#include "forward.h"
#include <math.h>
#include <stdio.h>

void mlp_forward(
    const float *input,
    int in_dim,
    int out_dim,
    const float *weight,
    const float *bias,
    float *output
) {
    for (int i = 0; i < out_dim; i++) {
        output[i] = bias[i];
        for (int j = 0; j < in_dim; j++) {
            output[i] += input[j] * weight[i * in_dim + j];
        }
        float x = output[i];
        output[i] = x * (1.0f / (1.0f + expf(-x)));  
    }
}

void encode_block_conditional(
    float ****input,
    float ****output_skip,
    const float *emb,
    int batch_size,
    int height,
    int width,
    int f0, int f1, int k0, int s0, int p0,
    int f2, int f3, int k1, int s1, int p1,
    const float *w0, const float *b0,
    const float *g0, const float *b_0, const float *m0, const float *v0,
    const float *w1, const float *b1,
    const float *g1, const float *b_1, const float *m1, const float *v1
) {
    float ****tmp = alloc_4d_arr(batch_size, height, width, f1);
    conv2d(input, tmp, w0, b0, batch_size, f0, f1, height, width, k0, s0, p0);
    batch_normalization(tmp, g0, b_0, m0, v0, batch_size, height, width, f1, 0.001f, 0.99f);
    silu(tmp, batch_size, height, width, f1);
    float ****mod = alloc_4d_arr(batch_size, height, width, f1);
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                for (int c = 0; c < f1; c++) {
                    mod[b][h][w][c] = tmp[b][h][w][c] * emb[c];
                }
            }
        }
    }
    conv2d(tmp, output_skip, w1, b1, batch_size, f1, f3, height, width, k1, s1, p1);
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                for (int c = 0; c < f3; c++) {
                    output_skip[b][h][w][c] += mod[b][h][w][c];
                }
            }
        }
    }
    batch_normalization(output_skip, g1, b_1, m1, v1, batch_size, height, width, f3, 0.001f, 0.99f);
    silu(output_skip, batch_size, height, width, f3);

    free_4d_arr(tmp,   batch_size, height, width);
    free_4d_arr(mod,   batch_size, height, width);
}

void decode_block_conditional(
    float ****in0,            // 이전 레벨 출력
    float ****in1,            // skip 연결 텐서
    float ****output,         // 이 레벨의 최종 출력
    const float *emb,         // 채널별 scale (조건부 임베딩)
    int batch_size,
    int height,               // in0의 H
    int width,                // in0의 W
    int f0, int f1, int k0, int s0, int p0,   // conv2d_transpose 파라미터
    int channels_skip,        // skip 텐서(in1)의 채널 수
    int f3, int k1, int s1, int p1,           // conv2d 파라미터
    const float *wT, const float *bT,
    const float *g0, const float *b_0, const float *m0, const float *v0,
    const float *w1, const float *b1,
    const float *g1, const float *b_1, const float *m1, const float *v1
) {
    // upsample 전: height×width, upsample 후: oh×ow
    int oh = height << 1;
    int ow = width  << 1;

    // debug
    printf("[DEBUG decode] batch=%d, in=(%d×%d), out=(%d×%d), f1=%d, skip_ch=%d, f3=%d\n",
           batch_size, height, width, oh, ow, f1, channels_skip, f3);

    // 1) Conv2DTranspose → tmp: [batch][oh][ow][f1]
    float ****tmp = alloc_4d_arr(batch_size, oh, ow, f1);
    conv2d_transpose(in0, tmp, wT, bT,
                     batch_size, f0, f1,
                     height, width, k0, s0, p0);

    // 2) BatchNorm + SiLU
    batch_normalization(tmp, g0, b_0, m0, v0,
                        batch_size, oh, ow, f1, 0.001f, 0.99f);
    silu(tmp, batch_size, oh, ow, f1);

    // 3) embedding 곱 → mod: [batch][oh][ow][f1]
    float ****mod = alloc_4d_arr(batch_size, oh, ow, f1);
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < oh*ow; i++) {
            int h = i / ow, w = i % ow;
            for (int c = 0; c < f1; c++) {
                mod[b][h][w][c] = tmp[b][h][w][c] * emb[c];
            }
        }
    }

    // 4) concat(tmp, in1) → cat: [batch][oh][ow][f1 + channels_skip]
    int total_ch = f1 + channels_skip;
    float ****cat = alloc_4d_arr(batch_size, oh, ow, total_ch);
    concatenate(tmp, in1, cat,
                batch_size, oh, ow,
                f1, channels_skip);

    // 5) Conv2D(cat) → output + residual(mod) → BatchNorm + SiLU
    conv2d(cat, output, w1, b1,
           batch_size, total_ch, f3,
           oh, ow, k1, s1, p1);
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < oh*ow; i++) {
            int h = i / ow, w = i % ow;
            for (int c = 0; c < f3; c++) {
                output[b][h][w][c] += mod[b][h][w][c];
            }
        }
    }
    batch_normalization(output, g1, b_1, m1, v1,
                        batch_size, oh, ow, f3, 0.001f, 0.99f);
    silu(output, batch_size, oh, ow, f3);

    // 6) 해제
    free_4d_arr(tmp, batch_size, oh, ow);
    free_4d_arr(mod, batch_size, oh, ow);
    free_4d_arr(cat, batch_size, oh, ow);
}

void encoding(
    float ****input,        
    float ****x0,           
    float ****x1,           
    float ****x2,           
    float ****x3,           
    float ****x4,          
    float **bc_input,
    int batch_size
) {
    float *bc_emb_128 = (float *)malloc(128 * sizeof(float));
    mlp_forward(
        bc_input[0],              
        UNET_PAR_0_DENSE_IN_DIM,  
        UNET_PAR_0_DENSE_OUT_DIM, 
        layer_0_dense_weight,     
        layer_0_dense_bias,       
        bc_emb_128                
    );
    float *bc_emb_20 = (float *)malloc(20 * sizeof(float));
    mlp_forward(
        bc_emb_128,               
        UNET_PAR_2_DENSE_IN_DIM,  
        UNET_PAR_2_DENSE_OUT_DIM, 
        layer_2_dense_weight,     
        layer_2_dense_bias,       
        bc_emb_20                 
    );

    encode_block_conditional(
        input, x0, bc_emb_20,
        batch_size,
        UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM,          
        UNET_PAR_1_CONV2D_IN_FILTER,   
        UNET_PAR_1_CONV2D_OUT_FILTER,  
        UNET_PAR_1_CONV2D_KERNEL_Y,    
        1, 1,
        UNET_PAR_4_CONV2D_IN_FILTER,   
        UNET_PAR_4_CONV2D_OUT_FILTER,  
        UNET_PAR_4_CONV2D_KERNEL_Y,    
        1, 1,
        layer_1_conv2d_weight, layer_1_conv2d_bias,
        layer_3_batch_norm_gamma, layer_3_batch_norm_beta,
        layer_3_batch_norm_mean, layer_3_batch_norm_variance,
        layer_4_conv2d_weight, layer_4_conv2d_bias,
        layer_5_batch_norm_gamma, layer_5_batch_norm_beta,
        layer_5_batch_norm_mean, layer_5_batch_norm_variance
    );
    printf("[SHAPE] x0 = [%d × %d × %d × %d]\n",
           batch_size,
           UNET_PAR_ROW_NUM,
           UNET_PAR_COL_NUM,
           UNET_PAR_4_CONV2D_OUT_FILTER);

    float *bc_emb_40 = (float *)malloc(40 * sizeof(float));
    mlp_forward(
        bc_emb_128,               
        UNET_PAR_7_DENSE_IN_DIM,  
        UNET_PAR_7_DENSE_OUT_DIM, 
        layer_7_dense_weight,     
        layer_7_dense_bias,       
        bc_emb_40                 
    );

    encode_block_conditional(
        x0, x1, bc_emb_40,
        batch_size,
        UNET_PAR_ROW_NUM >> 1, UNET_PAR_COL_NUM >> 1,  
        UNET_PAR_6_CONV2D_IN_FILTER,   
        UNET_PAR_6_CONV2D_OUT_FILTER,  
        UNET_PAR_6_CONV2D_KERNEL_Y,    
        1, 1,
        UNET_PAR_9_CONV2D_IN_FILTER,   
        UNET_PAR_9_CONV2D_OUT_FILTER,  
        UNET_PAR_9_CONV2D_KERNEL_Y,    
        1, 1,
        layer_6_conv2d_weight, layer_6_conv2d_bias,
        layer_8_batch_norm_gamma, layer_8_batch_norm_beta,
        layer_8_batch_norm_mean, layer_8_batch_norm_variance,
        layer_9_conv2d_weight, layer_9_conv2d_bias,
        layer_10_batch_norm_gamma, layer_10_batch_norm_beta,
        layer_10_batch_norm_mean, layer_10_batch_norm_variance
    );
    printf("[SHAPE] x1 = [%d × %d × %d × %d]\n",
           batch_size,
           UNET_PAR_ROW_NUM>>1,
           UNET_PAR_COL_NUM>>1,
           UNET_PAR_9_CONV2D_OUT_FILTER);
    float ****pooled_x0 = alloc_4d_arr(
        batch_size,
        UNET_PAR_ROW_NUM >> 1, UNET_PAR_COL_NUM >> 1,
        UNET_PAR_9_CONV2D_OUT_FILTER
    );
    max_pool(
        x1, pooled_x0, batch_size,
        UNET_PAR_ROW_NUM >> 1, UNET_PAR_COL_NUM >> 1,
        UNET_PAR_9_CONV2D_OUT_FILTER,
        2, 2
    );
    printf("[SHAPE] pooled_x0 = [%d × %d × %d × %d]\n",
           batch_size,
           UNET_PAR_ROW_NUM>>1,
           UNET_PAR_COL_NUM>>1,
           UNET_PAR_9_CONV2D_OUT_FILTER);

    float *bc_emb_80 = (float *)malloc(80 * sizeof(float));
    mlp_forward(
        bc_emb_128,               
        UNET_PAR_12_DENSE_IN_DIM,  
        UNET_PAR_12_DENSE_OUT_DIM, 
        layer_12_dense_weight,     
        layer_12_dense_bias,       
        bc_emb_80                  
    );

    encode_block_conditional(
        pooled_x0, x2, bc_emb_80,
        batch_size,
        (UNET_PAR_ROW_NUM >> 1) >> 1, (UNET_PAR_COL_NUM >> 1) >> 1,  
        UNET_PAR_11_CONV2D_IN_FILTER,   
        UNET_PAR_11_CONV2D_OUT_FILTER,  
        UNET_PAR_11_CONV2D_KERNEL_Y,    
        1, 1,
        UNET_PAR_14_CONV2D_IN_FILTER,   
        UNET_PAR_14_CONV2D_OUT_FILTER,  
        UNET_PAR_14_CONV2D_KERNEL_Y,    
        1, 1,
        layer_11_conv2d_weight, layer_11_conv2d_bias,
        layer_13_batch_norm_gamma, layer_13_batch_norm_beta,
        layer_13_batch_norm_mean, layer_13_batch_norm_variance,
        layer_14_conv2d_weight, layer_14_conv2d_bias,
        layer_15_batch_norm_gamma, layer_15_batch_norm_beta,
        layer_15_batch_norm_mean, layer_15_batch_norm_variance
    );
    printf("[SHAPE] x2 = [%d × %d × %d × %d]\n",
           batch_size,
           (UNET_PAR_ROW_NUM>>1)>>1,
           (UNET_PAR_COL_NUM>>1)>>1,
           UNET_PAR_14_CONV2D_OUT_FILTER);
    float ****pooled_x1 = alloc_4d_arr(
        batch_size,
        (UNET_PAR_ROW_NUM >> 1) >> 1, (UNET_PAR_COL_NUM >> 1) >> 1,
        UNET_PAR_14_CONV2D_OUT_FILTER
    );
    max_pool(
        x2, pooled_x1, batch_size,
        (UNET_PAR_ROW_NUM >> 1) >> 1, (UNET_PAR_COL_NUM >> 1) >> 1,
        UNET_PAR_14_CONV2D_OUT_FILTER,
        2, 2
    );
    printf("[SHAPE] pooled_x1 = [%d × %d × %d × %d]\n",
           batch_size,
           (UNET_PAR_ROW_NUM>>1)>>1,
           (UNET_PAR_COL_NUM>>1)>>1,
           UNET_PAR_14_CONV2D_OUT_FILTER);
    free_4d_arr(pooled_x0, batch_size, UNET_PAR_ROW_NUM >> 1, UNET_PAR_COL_NUM >> 1);
    
    float *bc_emb_160 = (float *)malloc(160 * sizeof(float));
    mlp_forward(
        bc_emb_128,               
        UNET_PAR_17_DENSE_IN_DIM,  
        UNET_PAR_17_DENSE_OUT_DIM, 
        layer_17_dense_weight,     
        layer_17_dense_bias,       
        bc_emb_160                 
    );

    encode_block_conditional(
        pooled_x1, x3, bc_emb_160,
        batch_size,
        UNET_PAR_ROW_NUM >> 3, UNET_PAR_COL_NUM >> 3, 
        UNET_PAR_14_CONV2D_IN_FILTER, 
        UNET_PAR_16_CONV2D_OUT_FILTER,
        UNET_PAR_16_CONV2D_KERNEL_Y,  
        1, 1,
        UNET_PAR_19_CONV2D_IN_FILTER,   
        UNET_PAR_19_CONV2D_OUT_FILTER,  
        UNET_PAR_19_CONV2D_KERNEL_Y,    
        1, 1,
        layer_16_conv2d_weight, layer_16_conv2d_bias,
        layer_18_batch_norm_gamma, layer_18_batch_norm_beta,
        layer_18_batch_norm_mean, layer_18_batch_norm_variance,
        layer_19_conv2d_weight, layer_19_conv2d_bias,
        layer_20_batch_norm_gamma, layer_20_batch_norm_beta,
        layer_20_batch_norm_mean, layer_20_batch_norm_variance
    );
    printf("[SHAPE] x3 = [%d × %d × %d × %d]\n",
           batch_size,
           UNET_PAR_ROW_NUM>>3,
           UNET_PAR_COL_NUM>>3,
           UNET_PAR_19_CONV2D_OUT_FILTER);
    float ****pooled_x2 = alloc_4d_arr(
        batch_size,
        ((UNET_PAR_ROW_NUM >> 1) >> 1) >> 1, ((UNET_PAR_COL_NUM >> 1) >> 1) >> 1,
        UNET_PAR_19_CONV2D_OUT_FILTER
    );
    max_pool(
        x3, pooled_x2, batch_size,
        ((UNET_PAR_ROW_NUM >> 1) >> 1) >> 1, ((UNET_PAR_COL_NUM >> 1) >> 1) >> 1,
        UNET_PAR_19_CONV2D_OUT_FILTER,
        2, 2
    );
    printf("[SHAPE] pooled_x2 = [%d × %d × %d × %d]\n",
           batch_size,
           ((UNET_PAR_ROW_NUM>>1)>>1)>>1,
           ((UNET_PAR_COL_NUM>>1)>>1)>>1,
           UNET_PAR_19_CONV2D_OUT_FILTER);
    free_4d_arr(pooled_x1, batch_size, (UNET_PAR_ROW_NUM >> 1) >> 1, (UNET_PAR_COL_NUM >> 1) >> 1);
    
    float *bc_emb_320 = (float *)malloc(320 * sizeof(float));
    mlp_forward(
        bc_emb_128,               
        UNET_PAR_22_DENSE_IN_DIM,  
        UNET_PAR_22_DENSE_OUT_DIM, 
        layer_22_dense_weight,     
        layer_22_dense_bias,       
        bc_emb_320                 
    );

    encode_block_conditional(
        pooled_x2, x4, bc_emb_320,
        batch_size,
        UNET_PAR_ROW_NUM >> 3, UNET_PAR_COL_NUM >> 3,  
        UNET_PAR_19_CONV2D_OUT_FILTER,   
        UNET_PAR_21_CONV2D_OUT_FILTER,   
        UNET_PAR_21_CONV2D_KERNEL_Y,     
        1, 1,
        UNET_PAR_24_CONV2D_IN_FILTER,    
        UNET_PAR_24_CONV2D_OUT_FILTER,   
        UNET_PAR_24_CONV2D_KERNEL_Y,     
        1, 1,
        layer_21_conv2d_weight, layer_21_conv2d_bias,
        layer_23_batch_norm_gamma, layer_23_batch_norm_beta,
        layer_23_batch_norm_mean, layer_23_batch_norm_variance,
        layer_24_conv2d_weight, layer_24_conv2d_bias,
        layer_25_batch_norm_gamma, layer_25_batch_norm_beta,
        layer_25_batch_norm_mean, layer_25_batch_norm_variance
    );
    printf("[SHAPE] x4 = [%d × %d × %d × %d]\n",
           batch_size,
           UNET_PAR_ROW_NUM>>3,
           UNET_PAR_COL_NUM>>3,
           UNET_PAR_24_CONV2D_OUT_FILTER);
    free_4d_arr(pooled_x2, batch_size, ((UNET_PAR_ROW_NUM >> 1) >> 1) >> 1, ((UNET_PAR_COL_NUM >> 1) >> 1) >> 1);
    
    free(bc_emb_20);
    free(bc_emb_40);
    free(bc_emb_80);
    free(bc_emb_160);
    free(bc_emb_320);
    free(bc_emb_128);
}

void decoding(
    float ****bottleneck,   // x4, 16×12×320
    float ****output,       // 최종 output, 82×67×1
    float ****x0,           // skip-level0: 128×96×20
    float ****x1,           // skip-level1:  64×48×40
    float ****x2,           // skip-level2:  32×24×80
    float ****x3,           // skip-level3:  16×12×160
    float **bc_input,
    int batch_size
) {
    // -------------- D0 -----------------
    // 1) conditional embedding
    float *bc_emb_128 = malloc(128*sizeof(float));
    mlp_forward(bc_input[0],
                UNET_PAR_0_DENSE_IN_DIM, UNET_PAR_0_DENSE_OUT_DIM,
                layer_0_dense_weight, layer_0_dense_bias,
                bc_emb_128);
    float *bc_emb_160 = malloc(160*sizeof(float));
    mlp_forward(bc_emb_128,
                UNET_PAR_28_DENSE_IN_DIM, UNET_PAR_28_DENSE_OUT_DIM,
                layer_28_dense_weight, layer_28_dense_bias,
                bc_emb_160);
    // d0: 32×24×160
    float ****d0 = alloc_4d_arr(
        batch_size,
        (UNET_PAR_ROW_NUM>>3)<<1,   // 16→32
        (UNET_PAR_COL_NUM>>3)<<1,   // 12→24
        UNET_PAR_29_CONV2D_OUT_FILTER // 160
    );
    decode_block_conditional(
        bottleneck, x3, d0, bc_emb_160,
        batch_size,
        UNET_PAR_ROW_NUM>>3, UNET_PAR_COL_NUM>>3,         // 16,12
        UNET_PAR_26_CONV2D_TRANSPOSE_IN_FILTER,          // f0=320
        UNET_PAR_26_CONV2D_TRANSPOSE_OUT_FILTER,         // f1=160
        UNET_PAR_26_CONV2D_TRANSPOSE_KERNEL_Y, 2,1,       // k0,s0,p0
        UNET_PAR_19_CONV2D_IN_FILTER,                   // skip 채널=160
        UNET_PAR_29_CONV2D_OUT_FILTER,                  // f3=160
        UNET_PAR_29_CONV2D_KERNEL_Y, 1,1,                // k1,s1,p1
        layer_26_conv2d_transpose_weight,
        layer_26_conv2d_transpose_bias,
        layer_27_batch_norm_gamma, layer_27_batch_norm_beta,
        layer_27_batch_norm_mean,  layer_27_batch_norm_variance,
        layer_29_conv2d_weight,    layer_29_conv2d_bias,
        layer_30_batch_norm_gamma, layer_30_batch_norm_beta,
        layer_30_batch_norm_mean,  layer_30_batch_norm_variance
    );

    // -------------- D1 -----------------
    float *bc_emb_80 = malloc(80*sizeof(float));
    mlp_forward(bc_emb_128,
                UNET_PAR_33_DENSE_IN_DIM, UNET_PAR_33_DENSE_OUT_DIM,
                layer_33_dense_weight, layer_33_dense_bias,
                bc_emb_80);
    // d1: 64×48×80
    float ****d1 = alloc_4d_arr(
        batch_size,
        (UNET_PAR_ROW_NUM>>2)<<1,   // 32→64
        (UNET_PAR_COL_NUM>>2)<<1,   // 24→48
        UNET_PAR_34_CONV2D_OUT_FILTER // 80
    );
    decode_block_conditional(
        d0, x2, d1, bc_emb_80,
        batch_size,
        UNET_PAR_ROW_NUM>>2, UNET_PAR_COL_NUM>>2,         // 32,24
        UNET_PAR_31_CONV2D_TRANSPOSE_IN_FILTER,          // f0=160
        UNET_PAR_31_CONV2D_TRANSPOSE_OUT_FILTER,         // f1=80
        UNET_PAR_31_CONV2D_TRANSPOSE_KERNEL_Y, 2,1,
        UNET_PAR_14_CONV2D_OUT_FILTER,                   // skip 채널=80
        UNET_PAR_34_CONV2D_OUT_FILTER,                   // f3=80
        UNET_PAR_34_CONV2D_KERNEL_Y, 1,1,
        layer_31_conv2d_transpose_weight,
        layer_31_conv2d_transpose_bias,
        layer_32_batch_norm_gamma, layer_32_batch_norm_beta,
        layer_32_batch_norm_mean,  layer_32_batch_norm_variance,
        layer_34_conv2d_weight,    layer_34_conv2d_bias,
        layer_35_batch_norm_gamma, layer_35_batch_norm_beta,
        layer_35_batch_norm_mean,  layer_35_batch_norm_variance
    );
    free_4d_arr(d0, batch_size,
                UNET_PAR_ROW_NUM>>2, UNET_PAR_COL_NUM>>2);

    // -------------- D2 -----------------
    float *bc_emb_40 = malloc(40*sizeof(float));
    mlp_forward(bc_emb_128,
                UNET_PAR_38_DENSE_IN_DIM, UNET_PAR_38_DENSE_OUT_DIM,
                layer_38_dense_weight, layer_38_dense_bias,
                bc_emb_40);
    // d2: 128×96×40
    float ****d2 = alloc_4d_arr(
        batch_size,
        (UNET_PAR_ROW_NUM>>1)<<1,   // 64→128
        (UNET_PAR_COL_NUM>>1)<<1,   // 48→96
        UNET_PAR_39_CONV2D_OUT_FILTER // 40
    );
    decode_block_conditional(
        d1, x1, d2, bc_emb_40,
        batch_size,
        UNET_PAR_ROW_NUM>>1, UNET_PAR_COL_NUM>>1,         // 64,48
        UNET_PAR_36_CONV2D_TRANSPOSE_IN_FILTER,          // f0=80
        UNET_PAR_36_CONV2D_TRANSPOSE_OUT_FILTER,         // f1=40
        UNET_PAR_36_CONV2D_TRANSPOSE_KERNEL_Y, 2,1,
        UNET_PAR_9_CONV2D_OUT_FILTER,                    // skip 채널=40
        UNET_PAR_39_CONV2D_OUT_FILTER,                   // f3=40
        UNET_PAR_39_CONV2D_KERNEL_Y, 1,1,
        layer_36_conv2d_transpose_weight,
        layer_36_conv2d_transpose_bias,
        layer_37_batch_norm_gamma, layer_37_batch_norm_beta,
        layer_37_batch_norm_mean,  layer_37_batch_norm_variance,
        layer_39_conv2d_weight,    layer_39_conv2d_bias,
        layer_40_batch_norm_gamma, layer_40_batch_norm_beta,
        layer_40_batch_norm_mean,  layer_40_batch_norm_variance
    );
    free_4d_arr(d1, batch_size,
                UNET_PAR_ROW_NUM>>1, UNET_PAR_COL_NUM>>1);

    // -------------- Final resize & conv -----------------
    float ****d3_temp = alloc_4d_arr(
        batch_size,
        UNET_PAR_ROW_NUM,    // 128
        UNET_PAR_COL_NUM,    // 96
        UNET_PAR_39_CONV2D_OUT_FILTER // 40
    );
    resize_bilinear_batch(
        d2, d3_temp, batch_size,
        UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM,
        CMAQ_PAR_ROW_NUM, CMAQ_PAR_COL_NUM,
        UNET_PAR_39_CONV2D_OUT_FILTER
    );
    free_4d_arr(d2, batch_size,
                UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM);

    conv2d(d3_temp, output,
           layer_41_conv2d_weight, layer_41_conv2d_bias,
           batch_size,
           UNET_PAR_41_CONV2D_IN_FILTER,  // 40
           UNET_PAR_41_CONV2D_OUT_FILTER, // 1
           CMAQ_PAR_ROW_NUM, CMAQ_PAR_COL_NUM,
           1,1,0);

    // 절댓값
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < CMAQ_PAR_ROW_NUM * CMAQ_PAR_COL_NUM; i++) {
            int h = i / CMAQ_PAR_COL_NUM, w = i % CMAQ_PAR_COL_NUM;
            output[b][h][w][0] = fabsf(output[b][h][w][0]);
        }
    }
    free_4d_arr(d3_temp, batch_size,
                CMAQ_PAR_ROW_NUM, CMAQ_PAR_COL_NUM);

    free(bc_emb_128);
    free(bc_emb_160);
    free(bc_emb_80);
    free(bc_emb_40);
}
void forward(
    float ***input,
    float **bc_input,
    float ****output,
    int batch_size
) {
    float ****x0 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM, UNET_PAR_4_CONV2D_OUT_FILTER);  
    float ****x1 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM >> 1, UNET_PAR_COL_NUM >> 1, UNET_PAR_9_CONV2D_OUT_FILTER);   
    float ****x2 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM >> 2, UNET_PAR_COL_NUM >> 2, UNET_PAR_14_CONV2D_OUT_FILTER);  
    float ****x3 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM >> 3, UNET_PAR_COL_NUM >> 3, UNET_PAR_19_CONV2D_OUT_FILTER);  
    float ****x4 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM >> 3, UNET_PAR_COL_NUM >> 3, UNET_PAR_24_CONV2D_OUT_FILTER);  

    float ****gridded_input = alloc_4d_arr(batch_size, CMAQ_PAR_ROW_NUM, CMAQ_PAR_COL_NUM, CMAQ_PAR_SECTOR_NUM);
    gridding(input, gridded_input, batch_size, CMAQ_PAR_REGION_NUM, CMAQ_PAR_SECTOR_NUM);

    float ****resized_gridded_input = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM, CMAQ_PAR_SECTOR_NUM);
    resize_bilinear_batch(gridded_input, resized_gridded_input, batch_size, CMAQ_PAR_ROW_NUM, CMAQ_PAR_COL_NUM, UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM, CMAQ_PAR_SECTOR_NUM);

    encoding(resized_gridded_input, x0, x1, x2, x3, x4, bc_input, batch_size);
    decoding(x4, output, x0, x1, x2, x3, bc_input, batch_size);
    
    free_4d_arr(gridded_input, batch_size, CMAQ_PAR_ROW_NUM, CMAQ_PAR_COL_NUM);
    free_4d_arr(resized_gridded_input, batch_size, UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM);
    free_4d_arr(x0, batch_size, UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM);
    free_4d_arr(x1, batch_size, UNET_PAR_ROW_NUM >> 1, UNET_PAR_COL_NUM >> 1);
    free_4d_arr(x2, batch_size, UNET_PAR_ROW_NUM >> 2, UNET_PAR_COL_NUM >> 2);
    free_4d_arr(x3, batch_size, UNET_PAR_ROW_NUM >> 2, UNET_PAR_COL_NUM >> 2);
    free_4d_arr(x4, batch_size, UNET_PAR_ROW_NUM >> 3, UNET_PAR_COL_NUM >> 3);
}

void call(
    float *c_inputs,
    float *c_bc_inputs,
    float *c_outputs,
    int batch_size
) {
    
    float ***input = malloc(batch_size * sizeof(float **));
    float **bc_input = malloc(batch_size * sizeof(float *));
    for (int b = 0; b < batch_size; b++) {
        input[b] = malloc(CMAQ_PAR_REGION_NUM * sizeof(float *));
        for (int i = 0; i < CMAQ_PAR_REGION_NUM; i++) {
            input[b][i] = malloc(CMAQ_PAR_SECTOR_NUM * sizeof(float));
            for (int s = 0; s < CMAQ_PAR_SECTOR_NUM; s++) {
                int idx = b * (CMAQ_PAR_REGION_NUM * CMAQ_PAR_SECTOR_NUM) + i * CMAQ_PAR_SECTOR_NUM + s;
                input[b][i][s] = c_inputs[idx];
            }
        }
        bc_input[b] = malloc(sizeof(float));
        bc_input[b][0] = c_bc_inputs[b];
    }
    float ****output = alloc_4d_arr(batch_size, CMAQ_PAR_ROW_NUM, CMAQ_PAR_COL_NUM, 1);

    forward(input, bc_input, output, batch_size);

    for (int b = 0; b < batch_size; b++) {
        printf("=== Batch %d output (82×67) ===\n", b);
        for (int h = 0; h < CMAQ_PAR_ROW_NUM; h++) {
            for (int w = 0; w < CMAQ_PAR_COL_NUM; w++) {
                printf("%f ", output[b][h][w][0]);
            }
            printf("\n");
        }
        printf("\n");  
    }
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < CMAQ_PAR_ROW_NUM; h++) {
            for (int w = 0; w < CMAQ_PAR_COL_NUM; w++) {
                int out_idx = b * (CMAQ_PAR_ROW_NUM * CMAQ_PAR_COL_NUM)
                              + h * CMAQ_PAR_COL_NUM + w;
                c_outputs[out_idx] = output[b][h][w][0];
            }
        }
    }

    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < CMAQ_PAR_REGION_NUM; i++) {
            free(input[b][i]);
        }
        free(input[b]);
        free(bc_input[b]);
    }
    free(input);
    free(bc_input);
    free_4d_arr(output, batch_size, CMAQ_PAR_ROW_NUM, CMAQ_PAR_COL_NUM);
}

float *set_sector(void) {
    float *sector = (float *)malloc(CMAQ_PAR_SECTOR_NUM * sizeof(float));
    scanf("%f %f %f %f %f", &sector[0], &sector[1], &sector[2], &sector[3], &sector[4]);
    return sector;
}

float *set_bc(void) {
    float *bc = (float *)malloc(sizeof(float));
    scanf("%f", bc);
    return bc;
}

float **set_bc_mat(int batch_size) {
    float **bc_mat = (float **)malloc(batch_size * sizeof(float *));
    for (int b = 0; b < batch_size; b++) {
        bc_mat[b] = set_bc();
    }
    return bc_mat;
}

float ***set_ctrl_mat(int batch_size) {
    float ***ctrl_mat = (float ***)malloc(batch_size * sizeof(float **));
    for (int b = 0; b < batch_size; b++) {
        ctrl_mat[b] = (float **)malloc(CMAQ_PAR_REGION_NUM * sizeof(float *));
        for (int i = 0; i < CMAQ_PAR_REGION_NUM; i++) {
            ctrl_mat[b][i] = set_sector();
        }
    }
    return ctrl_mat;
}

int main(void) {
    int batch_size = 1;
    
    float ***inputs = set_ctrl_mat(batch_size);
    float **bc_inputs = set_bc_mat(batch_size);
    float ****outputs = alloc_4d_arr(batch_size, CMAQ_PAR_ROW_NUM, CMAQ_PAR_COL_NUM, 1);
    
    forward(inputs, bc_inputs, outputs, batch_size);
    
    for (int b = 0; b < batch_size; b++) {
        printf("=== Batch %d output (82×67) ===\n", b);
        for (int h = 0; h < CMAQ_PAR_ROW_NUM; h++) {
            for (int w = 0; w < CMAQ_PAR_COL_NUM; w++) {
                printf("%f ", outputs[b][h][w][0]);
            }
            printf("\n");
        }
        printf("\n");
    }
    
    free_3d_arr(inputs, batch_size, CMAQ_PAR_REGION_NUM);
    for (int b = 0; b < batch_size; b++) {
        free(bc_inputs[b]);
    }
    free(bc_inputs);
    free_4d_arr(outputs, batch_size, CMAQ_PAR_ROW_NUM, CMAQ_PAR_COL_NUM);

    return 0;
}