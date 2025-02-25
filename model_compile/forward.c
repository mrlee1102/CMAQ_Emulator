#include "forward.h"

/*----------------------------------------------------------
 * 1) 인코더 블록: encode_block
 *    (Conv2D → BN → Activation → Conv2D → BN → Activation → MaxPool)
 *----------------------------------------------------------*/
// ==================== encode_block ====================
void encode_block(
    float ****input, float ****output_0, float ****output_1,
    int batch_size, int height, int width,
    int filter_0, int filter_1, int kernel_0, int stride_0, int padding_0,
    int filter_2, int filter_3, int kernel_1, int stride_1, int padding_1,
    int filter_4, int filter_5, int kernel_2, int stride_2, int padding_2,
    const float *weight_0, const float *bias_0,
    const float *weight_1, const float *bias_1,
    const float *gamma_0, const float *beta_0, const float *mean_0, const float *var_0,
    const float *weight_2, const float *bias_2,
    const float *gamma_1, const float *beta_1, const float *mean_1, const float *var_1
)
{
    // 임시 버퍼: (batch, height, width, filter_1)
    float ****tmp_0 = alloc_4d_arr(batch_size, height, width, filter_1);
    
    conv2d(input, tmp_0, weight_0, bias_0, batch_size,
           filter_0, filter_1, height, width, kernel_0, stride_0, padding_0);
    
    conv2d(tmp_0, tmp_0, weight_1, bias_1, batch_size,
           filter_2, filter_3, height, width, kernel_1, stride_1, padding_1);
    
    batch_normalization(tmp_0, gamma_0, beta_0, mean_0, var_0,
                        batch_size, height, width, filter_3, 0.001, 0.99);
    silu(tmp_0, batch_size, height, width, filter_3);
    
    conv2d(tmp_0, output_1, weight_2, bias_2, batch_size,
           filter_4, filter_5, height, width, kernel_2, stride_2, padding_2);
    
    batch_normalization(output_1, gamma_1, beta_1, mean_1, var_1,
                        batch_size, height, width, filter_5, 0.001, 0.99);
    silu(output_1, batch_size, height, width, filter_5);
    
    max_pool(output_1, output_0, batch_size, height, width, filter_5, 2, 2);
    
    free_4d_arr(tmp_0, batch_size, height, width);
}

// ==================== decode_block ====================
// 수정 포인트: (1) tmp_1 할당 시, 채널 수를 2*filter_1로 수정 (즉, 두 입력의 채널 합)
//             (2) 호출부에서 전달되는 height, width는 pre‑업샘플 크기 (예: 첫 디코더는 (6,4))
void decode_block(
    float ****input_0, float ****input_1, float ****output,
    int batch_size, int height, int width,
    // [A] Conv2DTranspose 파라미터
    int filter_0, int filter_1, int kernel_0, int stride_0, int padding_0,
    // [B] 첫 번째 Conv2D (Concat 후) 파라미터
    int filter_2, int filter_3, int kernel_1, int stride_1, int padding_1,
    // [C] 두 번째 Conv2D 파라미터
    int filter_4, int filter_5, int kernel_2, int stride_2, int padding_2,
    // [D] Conv2DTranspose 가중치/편향
    const float *weight_0, const float *bias_0,
    // [E] BN (Conv2DTranspose 후) 파라미터
    const float *gamma_0, const float *beta_0, const float *mean_0, const float *var_0,
    // [F] 첫 번째 Conv2D 가중치/편향 (Concat 후)
    const float *weight_1, const float *bias_1,
    // [H] 두 번째 Conv2D 가중치/편향
    const float *weight_2, const float *bias_2,
    // [G] BN (첫 번째 Conv2D 후) 파라미터
    const float *gamma_1, const float *beta_1, const float *mean_1, const float *var_1
)
{
    // tmp_0: Conv2DTranspose 결과 → (batch, height<<1, width<<1, filter_1)
    float ****tmp_0 = alloc_4d_arr(batch_size, height << 1, width << 1, filter_1);
    
    // tmp_1: Concat 결과, 두 입력 모두 채널 수 = filter_1이므로 총 채널 수 = 2 * filter_1
    float ****tmp_1 = alloc_4d_arr(batch_size, height << 1, width << 1, filter_1 * 2);
    
    // tmp_2: 첫 번째 Conv2D 결과 후, 채널 수 = filter_3
    float ****tmp_2 = alloc_4d_arr(batch_size, height << 1, width << 1, filter_3);
    
    conv2d_transpose(input_0, tmp_0, weight_0, bias_0, batch_size,
                     filter_0, filter_1, height, width, kernel_0, stride_0, padding_0);
    
    batch_normalization(tmp_0, gamma_0, beta_0, mean_0, var_0,
                        batch_size, height << 1, width << 1, filter_1, 0.001, 0.99);
    silu(tmp_0, batch_size, height << 1, width << 1, filter_1);
    
    concatenate(tmp_0, input_1, tmp_1, batch_size, height << 1, width << 1, filter_1, filter_1);
    
    conv2d(tmp_1, tmp_2, weight_1, bias_1, batch_size,
           filter_1 * 2, filter_3, height << 1, width << 1, kernel_1, stride_1, padding_1);
    
    conv2d(tmp_2, output, weight_2, bias_2, batch_size,
           filter_4, filter_5, height << 1, width << 1, kernel_2, stride_2, padding_2);
    
    batch_normalization(output, gamma_1, beta_1, mean_1, var_1,
                        batch_size, height << 1, width << 1, filter_5, 0.001, 0.99);
    silu(output, batch_size, height << 1, width << 1, filter_5);
    
    free_4d_arr(tmp_0, batch_size, height << 1, width << 1);
    free_4d_arr(tmp_1, batch_size, height << 1, width << 1);
    free_4d_arr(tmp_2, batch_size, height << 1, width << 1);
}

// ==================== encoding ====================
void encoding(float ****input, float ****x0, float ****x1, float ****x2, float ****x3, float ****x4, int batch_size)
{
    float ****tmp0 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM >> 1, UNET_PAR_COL_NUM >> 1, UNET_PAR_3_CONV2D_OUT_FILTER);
    float ****tmp1 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM >> 2, UNET_PAR_COL_NUM >> 2, UNET_PAR_8_CONV2D_OUT_FILTER);
    float ****tmp2 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM >> 3, UNET_PAR_COL_NUM >> 3, UNET_PAR_13_CONV2D_OUT_FILTER);
    float ****tmp3 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM >> 4, UNET_PAR_COL_NUM >> 4, UNET_PAR_18_CONV2D_OUT_FILTER);
    float ****tmp4 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM >> 5, UNET_PAR_COL_NUM >> 5, UNET_PAR_23_CONV2D_OUT_FILTER);

    // 인코더 블록 #1: (96,64,?) → x0: (96,64,UNET_PAR_3_CONV2D_OUT_FILTER)
    encode_block(
        input, tmp0, x0,
        batch_size, UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM,
        UNET_PAR_0_CONV2D_IN_FILTER, UNET_PAR_0_CONV2D_OUT_FILTER, UNET_PAR_0_CONV2D_KERNEL_Y, 1, 1,
        UNET_PAR_1_CONV2D_IN_FILTER, UNET_PAR_1_CONV2D_OUT_FILTER, UNET_PAR_1_CONV2D_KERNEL_Y, 1, 1,
        UNET_PAR_3_CONV2D_IN_FILTER, UNET_PAR_3_CONV2D_OUT_FILTER, UNET_PAR_3_CONV2D_KERNEL_Y, 1, 1,
        (float *)layer_0_conv2d_weight, (float *)layer_0_conv2d_bias,
        (float *)layer_1_conv2d_weight, (float *)layer_1_conv2d_bias,
        (float *)layer_2_batch_normalization_gamma, (float *)layer_2_batch_normalization_beta, (float *)layer_2_batch_normalization_mean, (float *)layer_2_batch_normalization_variance,
        (float *)layer_3_conv2d_weight, (float *)layer_3_conv2d_bias,
        (float *)layer_4_batch_normalization_gamma, (float *)layer_4_batch_normalization_beta, (float *)layer_4_batch_normalization_mean, (float *)layer_4_batch_normalization_variance
    );

    // 인코더 블록 #2: (48,32,?) → x1: (48,32,UNET_PAR_8_CONV2D_OUT_FILTER)
    encode_block(
        tmp0, tmp1, x1,
        batch_size, UNET_PAR_ROW_NUM >> 1, UNET_PAR_COL_NUM >> 1,
        UNET_PAR_5_CONV2D_IN_FILTER, UNET_PAR_5_CONV2D_OUT_FILTER, UNET_PAR_5_CONV2D_KERNEL_Y, 1, 1,
        UNET_PAR_6_CONV2D_IN_FILTER, UNET_PAR_6_CONV2D_OUT_FILTER, UNET_PAR_6_CONV2D_KERNEL_Y, 1, 1,
        UNET_PAR_8_CONV2D_IN_FILTER, UNET_PAR_8_CONV2D_OUT_FILTER, UNET_PAR_8_CONV2D_KERNEL_Y, 1, 1,
        (float *)layer_5_conv2d_weight, (float *)layer_5_conv2d_bias,
        (float *)layer_6_conv2d_weight, (float *)layer_6_conv2d_bias,
        (float *)layer_7_batch_normalization_gamma, (float *)layer_7_batch_normalization_beta, (float *)layer_7_batch_normalization_mean, (float *)layer_7_batch_normalization_variance,
        (float *)layer_8_conv2d_weight, (float *)layer_8_conv2d_bias,
        (float *)layer_9_batch_normalization_gamma, (float *)layer_9_batch_normalization_beta, (float *)layer_9_batch_normalization_mean, (float *)layer_9_batch_normalization_variance
    );

    // 인코더 블록 #3: (24,16,?) → x2: (24,16,UNET_PAR_13_CONV2D_OUT_FILTER)
    encode_block(
        tmp1, tmp2, x2,
        batch_size, UNET_PAR_ROW_NUM >> 2, UNET_PAR_COL_NUM >> 2,
        UNET_PAR_10_CONV2D_IN_FILTER, UNET_PAR_10_CONV2D_OUT_FILTER, UNET_PAR_10_CONV2D_KERNEL_Y, 1, 1,
        UNET_PAR_11_CONV2D_IN_FILTER, UNET_PAR_11_CONV2D_OUT_FILTER, UNET_PAR_11_CONV2D_KERNEL_Y, 1, 1,
        UNET_PAR_13_CONV2D_IN_FILTER, UNET_PAR_13_CONV2D_OUT_FILTER, UNET_PAR_13_CONV2D_KERNEL_Y, 1, 1,
        (float *)layer_10_conv2d_weight, (float *)layer_10_conv2d_bias,
        (float *)layer_11_conv2d_weight, (float *)layer_11_conv2d_bias,
        (float *)layer_12_batch_normalization_gamma, (float *)layer_12_batch_normalization_beta, (float *)layer_12_batch_normalization_mean, (float *)layer_12_batch_normalization_variance,
        (float *)layer_13_conv2d_weight, (float *)layer_13_conv2d_bias,
        (float *)layer_14_batch_normalization_gamma, (float *)layer_14_batch_normalization_beta, (float *)layer_14_batch_normalization_mean, (float *)layer_14_batch_normalization_variance
    );

    // 인코더 블록 #4: (12,8,?) → x3: (12,8,UNET_PAR_18_CONV2D_OUT_FILTER)
    encode_block(
        tmp2, tmp3, x3,
        batch_size, UNET_PAR_ROW_NUM >> 3, UNET_PAR_COL_NUM >> 3,
        UNET_PAR_15_CONV2D_IN_FILTER, UNET_PAR_15_CONV2D_OUT_FILTER, UNET_PAR_15_CONV2D_KERNEL_Y, 1, 1,
        UNET_PAR_16_CONV2D_IN_FILTER, UNET_PAR_16_CONV2D_OUT_FILTER, UNET_PAR_16_CONV2D_KERNEL_Y, 1, 1,
        UNET_PAR_18_CONV2D_IN_FILTER, UNET_PAR_18_CONV2D_OUT_FILTER, UNET_PAR_18_CONV2D_KERNEL_Y, 1, 1,
        (float *)layer_15_conv2d_weight, (float *)layer_15_conv2d_bias,
        (float *)layer_16_conv2d_weight, (float *)layer_16_conv2d_bias,
        (float *)layer_17_batch_normalization_gamma, (float *)layer_17_batch_normalization_beta, (float *)layer_17_batch_normalization_mean, (float *)layer_17_batch_normalization_variance,
        (float *)layer_18_conv2d_weight, (float *)layer_18_conv2d_bias,
        (float *)layer_19_batch_normalization_gamma, (float *)layer_19_batch_normalization_beta, (float *)layer_19_batch_normalization_mean, (float *)layer_19_batch_normalization_variance
    );

    // 인코더 블록 #5: (6,4,?) → x4: (6,4,UNET_PAR_23_CONV2D_OUT_FILTER)
    encode_block(
        tmp3, tmp4, x4,
        batch_size, UNET_PAR_ROW_NUM >> 4, UNET_PAR_COL_NUM >> 4,
        UNET_PAR_20_CONV2D_IN_FILTER, UNET_PAR_20_CONV2D_OUT_FILTER, UNET_PAR_20_CONV2D_KERNEL_Y, 1, 1,
        UNET_PAR_21_CONV2D_IN_FILTER, UNET_PAR_21_CONV2D_OUT_FILTER, UNET_PAR_21_CONV2D_KERNEL_Y, 1, 1,
        UNET_PAR_23_CONV2D_IN_FILTER, UNET_PAR_23_CONV2D_OUT_FILTER, UNET_PAR_23_CONV2D_KERNEL_Y, 1, 1,
        (float *)layer_20_conv2d_weight, (float *)layer_20_conv2d_bias,
        (float *)layer_21_conv2d_weight, (float *)layer_21_conv2d_bias,
        (float *)layer_22_batch_normalization_gamma, (float *)layer_22_batch_normalization_beta, (float *)layer_22_batch_normalization_mean, (float *)layer_22_batch_normalization_variance,
        (float *)layer_23_conv2d_weight, (float *)layer_23_conv2d_bias,
        (float *)layer_24_batch_normalization_gamma, (float *)layer_24_batch_normalization_beta, (float *)layer_24_batch_normalization_mean, (float *)layer_24_batch_normalization_variance
    );

    free_4d_arr(tmp0, batch_size, UNET_PAR_ROW_NUM >> 1, UNET_PAR_COL_NUM >> 1);
    free_4d_arr(tmp1, batch_size, UNET_PAR_ROW_NUM >> 2, UNET_PAR_COL_NUM >> 2);
    free_4d_arr(tmp2, batch_size, UNET_PAR_ROW_NUM >> 3, UNET_PAR_COL_NUM >> 3);
    free_4d_arr(tmp3, batch_size, UNET_PAR_ROW_NUM >> 4, UNET_PAR_COL_NUM >> 4);
    free_4d_arr(tmp4, batch_size, UNET_PAR_ROW_NUM >> 5, UNET_PAR_COL_NUM >> 5);
    return;
}

// ==================== decoding ====================
void decoding(float ****input, float ****output, float ****x0, float ****x1, float ****x2, float ****x3, int batch_size)
{
    float ****tmp0 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM >> 3, UNET_PAR_COL_NUM >> 3, UNET_PAR_28_CONV2D_OUT_FILTER);
    float ****tmp1 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM >> 2, UNET_PAR_COL_NUM >> 2, UNET_PAR_33_CONV2D_OUT_FILTER);
    float ****tmp2 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM >> 1, UNET_PAR_COL_NUM >> 1, UNET_PAR_38_CONV2D_OUT_FILTER);
    float ****tmp3 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM, UNET_PAR_43_CONV2D_OUT_FILTER);
    float ****tmp4 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM, UNET_PAR_45_CONV2D_OUT_FILTER);

    // 디코더 블록 #1: pre-upsample (6,4) → after Conv2DTranspose: (12,8)
    decode_block(
        input, x3, tmp0,
        batch_size, UNET_PAR_ROW_NUM >> 4, UNET_PAR_COL_NUM >> 4,  // (6,4)
        UNET_PAR_25_CONV2D_TRANSPOSE_IN_FILTER, UNET_PAR_25_CONV2D_TRANSPOSE_OUT_FILTER, UNET_PAR_25_CONV2D_TRANSPOSE_KERNEL_Y, 2, 0,
        UNET_PAR_27_CONV2D_IN_FILTER, UNET_PAR_27_CONV2D_OUT_FILTER, UNET_PAR_27_CONV2D_KERNEL_Y, 1, 1,
        UNET_PAR_28_CONV2D_IN_FILTER, UNET_PAR_28_CONV2D_OUT_FILTER, UNET_PAR_28_CONV2D_KERNEL_Y, 1, 1,
        (float *)layer_25_conv2d_transpose_weight, (float *)layer_25_conv2d_transpose_bias,
        (float *)layer_26_batch_normalization_gamma, (float *)layer_26_batch_normalization_beta, (float *)layer_26_batch_normalization_mean, (float *)layer_26_batch_normalization_variance,
        (float *)layer_27_conv2d_weight, (float *)layer_27_conv2d_bias,
        (float *)layer_28_conv2d_weight, (float *)layer_28_conv2d_bias,
        (float *)layer_29_batch_normalization_gamma, (float *)layer_29_batch_normalization_beta, (float *)layer_29_batch_normalization_mean, (float *)layer_29_batch_normalization_variance
    );

    // 디코더 블록 #2: pre-upsample (12,8) → (24,16)
    decode_block(
        tmp0, x2, tmp1,
        // batch_size, UNET_PAR_ROW_NUM >> 4, UNET_PAR_COL_NUM >> 4,  // pre-upsample: (6,4) was used in block1; now block2 input should be (12,8)
        // 만약 첫 블록의 output is (12,8), pre-upsample for block2 should be (12,8) 
        // → 따라서 여기서는 UNET_PAR_ROW_NUM >> 3, UNET_PAR_COL_NUM >> 3 = (12,8) 사용
        batch_size, UNET_PAR_ROW_NUM >> 3, UNET_PAR_COL_NUM >> 3,
        UNET_PAR_30_CONV2D_TRANSPOSE_IN_FILTER, UNET_PAR_30_CONV2D_TRANSPOSE_OUT_FILTER, UNET_PAR_30_CONV2D_TRANSPOSE_KERNEL_Y, 2, 0,
        UNET_PAR_32_CONV2D_IN_FILTER, UNET_PAR_32_CONV2D_OUT_FILTER, UNET_PAR_32_CONV2D_KERNEL_Y, 1, 1,
        UNET_PAR_33_CONV2D_IN_FILTER, UNET_PAR_33_CONV2D_OUT_FILTER, UNET_PAR_33_CONV2D_KERNEL_Y, 1, 1,
        (float *)layer_30_conv2d_transpose_weight, (float *)layer_30_conv2d_transpose_bias,
        (float *)layer_31_batch_normalization_gamma, (float *)layer_31_batch_normalization_beta, (float *)layer_31_batch_normalization_mean, (float *)layer_31_batch_normalization_variance,
        (float *)layer_32_conv2d_weight, (float *)layer_32_conv2d_bias,
        (float *)layer_33_conv2d_weight, (float *)layer_33_conv2d_bias,
        (float *)layer_34_batch_normalization_gamma, (float *)layer_34_batch_normalization_beta, (float *)layer_34_batch_normalization_mean, (float *)layer_34_batch_normalization_variance
    );

    // 디코더 블록 #3: pre-upsample (24,16) → (48,32)
    decode_block(
        tmp1, x1, tmp2,
        batch_size, UNET_PAR_ROW_NUM >> 2, UNET_PAR_COL_NUM >> 2,  // (24,16)
        UNET_PAR_35_CONV2D_TRANSPOSE_IN_FILTER, UNET_PAR_35_CONV2D_TRANSPOSE_OUT_FILTER, UNET_PAR_35_CONV2D_TRANSPOSE_KERNEL_Y, 2, 0,
        UNET_PAR_37_CONV2D_IN_FILTER, UNET_PAR_37_CONV2D_OUT_FILTER, UNET_PAR_37_CONV2D_KERNEL_Y, 1, 1,
        UNET_PAR_38_CONV2D_IN_FILTER, UNET_PAR_38_CONV2D_OUT_FILTER, UNET_PAR_38_CONV2D_KERNEL_Y, 1, 1,
        (float *)layer_35_conv2d_transpose_weight, (float *)layer_35_conv2d_transpose_bias,
        (float *)layer_36_batch_normalization_gamma, (float *)layer_36_batch_normalization_beta, (float *)layer_36_batch_normalization_mean, (float *)layer_36_batch_normalization_variance,
        (float *)layer_37_conv2d_weight, (float *)layer_37_conv2d_bias,
        (float *)layer_38_conv2d_weight, (float *)layer_38_conv2d_bias,
        (float *)layer_39_batch_normalization_gamma, (float *)layer_39_batch_normalization_beta, (float *)layer_39_batch_normalization_mean, (float *)layer_39_batch_normalization_variance
    );

    // 디코더 블록 #4: pre-upsample (48,32) → (96,64)
    decode_block(
        tmp2, x0, tmp3,
        batch_size, UNET_PAR_ROW_NUM >> 1, UNET_PAR_COL_NUM >> 1,  // (48,32)
        UNET_PAR_40_CONV2D_TRANSPOSE_IN_FILTER, UNET_PAR_40_CONV2D_TRANSPOSE_OUT_FILTER, UNET_PAR_40_CONV2D_TRANSPOSE_KERNEL_Y, 2, 0,
        UNET_PAR_42_CONV2D_IN_FILTER, UNET_PAR_42_CONV2D_OUT_FILTER, UNET_PAR_42_CONV2D_KERNEL_Y, 1, 1,
        UNET_PAR_43_CONV2D_IN_FILTER, UNET_PAR_43_CONV2D_OUT_FILTER, UNET_PAR_43_CONV2D_KERNEL_Y, 1, 1,
        (float *)layer_40_conv2d_transpose_weight, (float *)layer_40_conv2d_transpose_bias,
        (float *)layer_41_batch_normalization_gamma, (float *)layer_41_batch_normalization_beta, (float *)layer_41_batch_normalization_mean, (float *)layer_41_batch_normalization_variance,
        (float *)layer_42_conv2d_weight, (float *)layer_42_conv2d_bias,
        (float *)layer_43_conv2d_weight, (float *)layer_43_conv2d_bias,
        (float *)layer_44_batch_normalization_gamma, (float *)layer_44_batch_normalization_beta, (float *)layer_44_batch_normalization_mean, (float *)layer_44_batch_normalization_variance
    );

    conv2d(
        tmp3, tmp4, (float *)layer_45_conv2d_weight, (float *)layer_45_conv2d_bias,
        batch_size, UNET_PAR_45_CONV2D_IN_FILTER, UNET_PAR_45_CONV2D_OUT_FILTER,
        UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM, UNET_PAR_45_CONV2D_KERNEL_Y, 1, 0
    );
    resize_bilinear_batch(
        tmp4, output, batch_size,
        UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM,
        CMAQ_PAR_ROW_NUM, CMAQ_PAR_COL_NUM,
        UNET_PAR_45_CONV2D_OUT_FILTER
    );

    free_4d_arr(tmp0, batch_size, UNET_PAR_ROW_NUM >> 3, UNET_PAR_COL_NUM >> 3);
    free_4d_arr(tmp1, batch_size, UNET_PAR_ROW_NUM >> 2, UNET_PAR_COL_NUM >> 2);
    free_4d_arr(tmp2, batch_size, UNET_PAR_ROW_NUM >> 1, UNET_PAR_COL_NUM >> 1);
    free_4d_arr(tmp3, batch_size, UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM);
    free_4d_arr(tmp4, batch_size, UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM);
    return;
}

/*----------------------------------------------------------
 * 5) forward: 인코딩 + 디코딩 전체 과정
 *----------------------------------------------------------*/
void forward(float ***input, float ****output, int batch_size) {
    float ****gridded_input = alloc_4d_arr(batch_size, CMAQ_PAR_ROW_NUM, CMAQ_PAR_COL_NUM, CMAQ_PAR_SECTOR_NUM);
    float ****resized_gridded_input = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM, CMAQ_PAR_SECTOR_NUM);
    gridding(input, gridded_input, batch_size, CMAQ_PAR_REGION_NUM, CMAQ_PAR_SECTOR_NUM);
    resize_bilinear_batch(gridded_input, resized_gridded_input, batch_size,
                          CMAQ_PAR_ROW_NUM, CMAQ_PAR_COL_NUM,
                          UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM,
                          CMAQ_PAR_SECTOR_NUM);

    float ****x0 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM, UNET_PAR_3_CONV2D_OUT_FILTER);
    float ****x1 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM >> 1, UNET_PAR_COL_NUM >> 1, UNET_PAR_8_CONV2D_OUT_FILTER);
    float ****x2 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM >> 2, UNET_PAR_COL_NUM >> 2, UNET_PAR_13_CONV2D_OUT_FILTER);
    float ****x3 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM >> 3, UNET_PAR_COL_NUM >> 3, UNET_PAR_18_CONV2D_OUT_FILTER);
    float ****x4 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM >> 4, UNET_PAR_COL_NUM >> 4, UNET_PAR_23_CONV2D_OUT_FILTER);
    encoding(resized_gridded_input, x0, x1, x2, x3, x4, batch_size);

    decoding(x4, output, x0, x1, x2, x3, batch_size);
    
    free_4d_arr(gridded_input, batch_size, CMAQ_PAR_ROW_NUM, CMAQ_PAR_COL_NUM);
    free_4d_arr(resized_gridded_input, batch_size, UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM);
    free_4d_arr(x0, batch_size, UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM);
    free_4d_arr(x1, batch_size, UNET_PAR_ROW_NUM >> 1, UNET_PAR_COL_NUM >> 1);
    free_4d_arr(x2, batch_size, UNET_PAR_ROW_NUM >> 2, UNET_PAR_COL_NUM >> 2);
    free_4d_arr(x3, batch_size, UNET_PAR_ROW_NUM >> 3, UNET_PAR_COL_NUM >> 3);
    free_4d_arr(x4, batch_size, UNET_PAR_ROW_NUM >> 4, UNET_PAR_COL_NUM >> 4);
    return;
}

/*----------------------------------------------------------
 * 6) call: C 스타일 래퍼 함수
 *----------------------------------------------------------*/
void call(float *c_inputs, float *c_outputs, int batch_size) {
    float ***inputs = alloc_3d_arr(batch_size, CMAQ_PAR_REGION_NUM, CMAQ_PAR_SECTOR_NUM);
    float ****outputs = alloc_4d_arr(batch_size, CMAQ_PAR_ROW_NUM, CMAQ_PAR_COL_NUM, UNET_PAR_45_CONV2D_OUT_FILTER);

    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < CMAQ_PAR_REGION_NUM; j++) {
            for (size_t k = 0; k < CMAQ_PAR_SECTOR_NUM; k++) {
                inputs[i][j][k] = c_inputs[i * CMAQ_PAR_REGION_NUM * CMAQ_PAR_SECTOR_NUM + j * CMAQ_PAR_SECTOR_NUM + k];
            }
        }
    }

    forward(inputs, outputs, batch_size);

    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < CMAQ_PAR_ROW_NUM; j++) {
            for (size_t k = 0; k < CMAQ_PAR_COL_NUM; k++) {
                for (size_t l = 0; l < UNET_PAR_45_CONV2D_OUT_FILTER; l++) {
                    c_outputs[i * CMAQ_PAR_ROW_NUM * CMAQ_PAR_COL_NUM * UNET_PAR_45_CONV2D_OUT_FILTER +
                             j * CMAQ_PAR_COL_NUM * UNET_PAR_45_CONV2D_OUT_FILTER +
                             k * UNET_PAR_45_CONV2D_OUT_FILTER + l] =
                        outputs[i][j][k][l];
                }
            }
        }
    }

    free_3d_arr(inputs, batch_size, CMAQ_PAR_REGION_NUM);
    free_4d_arr(outputs, batch_size, CMAQ_PAR_ROW_NUM, CMAQ_PAR_COL_NUM);
    return;
}

float *set_sector(void) {
    float *sector = (float *)malloc(CMAQ_PAR_SECTOR_NUM * sizeof(float));
    scanf("%f %f %f %f %f %f %f",
          &sector[0], &sector[1], &sector[2],
          &sector[3], &sector[4], &sector[5], &sector[6]);
    return sector;
}

float ***set_ctrl_mat(int batch_size) {
    const char *region_codes = "ABCDEFGHIJKLMNOPQ";
    float ***ctrl_mat = (float ***)malloc(batch_size * sizeof(float **));
    for (int b = 0; b < batch_size; b++) {
        ctrl_mat[b] = (float **)malloc(CMAQ_PAR_REGION_NUM * sizeof(float *));
        for (int i = 0; i < CMAQ_PAR_REGION_NUM; i++) {
            printf("Enter the %c sector values: ", region_codes[i]);
            ctrl_mat[b][i] = set_sector();
        }
    }
    return ctrl_mat;
}

int main(void) {
    int batch_size = 1;
    float ***inputs = set_ctrl_mat(batch_size);
    float ****outputs = alloc_4d_arr(batch_size, CMAQ_PAR_ROW_NUM, CMAQ_PAR_COL_NUM, 1);

    printf("Complete Input Setup... \nProgress forward...");
    forward(inputs, outputs, batch_size);

    free_3d_arr(inputs, batch_size, CMAQ_PAR_REGION_NUM);
    free_4d_arr(outputs, batch_size, CMAQ_PAR_ROW_NUM, CMAQ_PAR_COL_NUM);
    return 0;
}
