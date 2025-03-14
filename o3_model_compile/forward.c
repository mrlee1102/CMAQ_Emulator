#include "forward.h"

/* 
   encode_block: 두 번의 Conv2D + BatchNorm + SiLU 연산을 수행하여 특징 추출을 합니다.
   (내부에서는 max pooling을 수행하지 않고, caller에서 필요 시 외부에서 max pooling을 적용합니다.)
*/
void encode_block(
    float ****input,       // 입력 feature map
    float ****output_skip, // 두 번째 Conv2D 결과 (skip 연결에 사용될 feature map)
    int batch_size, int height, int width,
    int filter_0, int filter_1, int kernel_0, int stride_0, int padding_0,
    int filter_2, int filter_3, int kernel_1, int stride_1, int padding_1,
    const float *weight_0, const float *bias_0,
    const float *gamma_0, const float *beta_0, const float *mean_0, const float *var_0,
    const float *weight_1, const float *bias_1,
    const float *gamma_1, const float *beta_1, const float *mean_1, const float *var_1
)
{
    float ****tmp = alloc_4d_arr(batch_size, height, width, filter_1);
    conv2d(input, tmp, weight_0, bias_0,
           batch_size, filter_0, filter_1,
           height, width, kernel_0, stride_0, padding_0);
    batch_normalization(tmp, gamma_0, beta_0, mean_0, var_0,
                        batch_size, height, width, filter_1, 0.001, 0.99);
    silu(tmp, batch_size, height, width, filter_1);


    conv2d(tmp, output_skip, weight_1, bias_1,
           batch_size, filter_2, filter_3,
           height, width, kernel_1, stride_1, padding_1);
    batch_normalization(output_skip, gamma_1, beta_1, mean_1, var_1,
                        batch_size, height, width, filter_3, 0.001, 0.99);
    silu(output_skip, batch_size, height, width, filter_3);

    free_4d_arr(tmp, batch_size, height, width);
    return;
}


/*
   decode_block: Conv2DTranspose(upsample) → BatchNorm+SiLU → Concatenate(skip) →
                 Conv2D → BatchNorm+SiLU.
    - conv2d_transpose는 'same' 업샘플(커널=3, stride=2, padding=1)로 동작하며,
    - conv2d는 'same' 유지(커널=3, stride=1, padding=1)합니다.
*/
void decode_block(
    float ****input_0, float ****input_1, float ****output,
    int batch_size, int height, int width,
    int filter_0, int filter_1, int kernel_0, int stride_0, int padding_0,
    int filter_2, int filter_3, int kernel_1, int stride_1, int padding_1,
    const float *weight_0, const float *bias_0,
    const float *gamma_0, const float *beta_0, const float *mean_0, const float *var_0,
    const float *weight_1, const float *bias_1,
    const float *gamma_1, const float *beta_1, const float *mean_1, const float *var_1)
{
    float ****tmp = alloc_4d_arr(batch_size, height << 1, width << 1, filter_1);
    float ****tmp2 = alloc_4d_arr(batch_size, height << 1, width << 1, filter_2);

    conv2d_transpose(input_0, tmp, weight_0, bias_0,
                     batch_size, filter_0, filter_1,
                     height, width, kernel_0, stride_0, padding_0);
    batch_normalization(tmp, gamma_0, beta_0, mean_0, var_0,
                        batch_size, (height << 1), (width << 1), filter_1, 0.001, 0.99);
    silu(tmp, batch_size, (height << 1), (width << 1), filter_1);

    concatenate(tmp, input_1, tmp2,
                batch_size, (height << 1), (width << 1), filter_1, filter_1);

    conv2d(tmp2, output, weight_1, bias_1,
           batch_size, filter_2, filter_3,
           (height << 1), (width << 1), kernel_1, stride_1, padding_1);
    batch_normalization(output, gamma_1, beta_1, mean_1, var_1,
                        batch_size, (height << 1), (width << 1), filter_3, 0.001, 0.99);
    silu(output, batch_size, (height << 1), (width << 1), filter_3);

    free_4d_arr(tmp,  batch_size, (height << 1), (width << 1));
    free_4d_arr(tmp2, batch_size, (height << 1), (width << 1));
    return;
}

/*======================================================================
   인코딩 (encoding)
     - 블록 1: 입력 (96×64×2) → 2번의 Conv2D 연산 후 skip x0: (96×64×10)
              → **max pooling 미실행** (skip으로 바로 전달)
     - 블록 2: 입력: x0 (96×64×10) → 2번의 Conv2D 연산 후 x1: (96×64×20)
              → 이후 max pooling → (48×32×20) → 다음 블록 입력
     - 블록 3: 입력: max_pool된 x1 (48×32×20) → 2번의 Conv2D 연산 후 x2: (48×32×40)
              → max pooling → (24×16×40) → 다음 블록 입력
     - 블록 4 (Bottleneck Part A): 입력: max_pool된 x2 (24×16×40) → 2번의 Conv2D 연산 후 x3: (24×16×80)
              → max pooling → (12×8×80) → 다음 블록 입력
     - 블록 5 (Bottleneck Part B): 입력: max_pool된 x3 (12×8×80) → 2번의 Conv2D 연산 후 bottleneck x4: (12×8×160)
              → **max pooling 미실행**
======================================================================*/
void encoding(
    float ****input,
    float ****x0, float ****x1, float ****x2, float ****x3, float ****x4,
    int batch_size)
{
    //---------------------------------------------------
    // 1) Block 1: 입력 (96×64×2) -> 두 번의 Conv -> x0 (96×64×10)
    //---------------------------------------------------
    encode_block(
        /* input   */ input,
        /* output */ x0,
        batch_size, 
        UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM,
        UNET_PAR_0_CONV2D_IN_FILTER,  UNET_PAR_0_CONV2D_OUT_FILTER,  UNET_PAR_0_CONV2D_KERNEL_Y, 1, 1,
        UNET_PAR_2_CONV2D_IN_FILTER,  UNET_PAR_2_CONV2D_OUT_FILTER,  UNET_PAR_2_CONV2D_KERNEL_Y, 1, 1,
        (float *)layer_0_conv2d_weight, (float *)layer_0_conv2d_bias,
        (float *)layer_1_batch_norm_gamma, (float *)layer_1_batch_norm_beta, 
        (float *)layer_1_batch_norm_mean,  (float *)layer_1_batch_norm_variance,
        (float *)layer_2_conv2d_weight, (float *)layer_2_conv2d_bias,
        (float *)layer_3_batch_norm_gamma, (float *)layer_3_batch_norm_beta, 
        (float *)layer_3_batch_norm_mean,  (float *)layer_3_batch_norm_variance
    );
    // x0의 shape: (96×64×10)

    // Block1 결과 x0에 max_pool 적용 -> pooled_x0 shape: (48×32×10)
    float ****pooled_x0 = alloc_4d_arr(
        batch_size,
        /* height = 96/2 */ UNET_PAR_ROW_NUM >> 1,
        /* width  = 64/2 */ UNET_PAR_COL_NUM >> 1,
        /* channel=10   */ UNET_PAR_2_CONV2D_OUT_FILTER
    );
    max_pool(x0, pooled_x0, batch_size, UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM, UNET_PAR_2_CONV2D_OUT_FILTER, 2, 2);
    //---------------------------------------------------
    // 2) Block 2: 입력 pooled_x0 (48×32×10) -> 두 번의 Conv -> x1 (48×32×20)
    //---------------------------------------------------
    encode_block(
        /* input   */ pooled_x0,
        /* output */ x1,
        batch_size, 
        UNET_PAR_ROW_NUM >> 1, UNET_PAR_COL_NUM >> 1,
        UNET_PAR_4_CONV2D_IN_FILTER,  UNET_PAR_4_CONV2D_OUT_FILTER,  UNET_PAR_4_CONV2D_KERNEL_Y, 1, 1,
        UNET_PAR_6_CONV2D_IN_FILTER,  UNET_PAR_6_CONV2D_OUT_FILTER,  UNET_PAR_6_CONV2D_KERNEL_Y, 1, 1,
        (float *)layer_4_conv2d_weight, (float *)layer_4_conv2d_bias,
        (float *)layer_5_batch_norm_gamma, (float *)layer_5_batch_norm_beta, 
        (float *)layer_5_batch_norm_mean,  (float *)layer_5_batch_norm_variance,
        (float *)layer_6_conv2d_weight, (float *)layer_6_conv2d_bias,
        (float *)layer_7_batch_norm_gamma, (float *)layer_7_batch_norm_beta, 
        (float *)layer_7_batch_norm_mean,  (float *)layer_7_batch_norm_variance
    );
    // x1의 shape: (48×32×20)

    // Block2 결과 x1에 max_pool -> pooled_x1 shape: (24×16×20)
    float ****pooled_x1 = alloc_4d_arr(
        batch_size,
        (UNET_PAR_ROW_NUM >> 1) >> 1, // 48/2 = 24
        (UNET_PAR_COL_NUM >> 1) >> 1, // 32/2 = 16
        UNET_PAR_4_CONV2D_OUT_FILTER  // 20
    );
    max_pool(x1, pooled_x1, batch_size, (UNET_PAR_ROW_NUM >> 1), (UNET_PAR_COL_NUM >> 1), UNET_PAR_4_CONV2D_OUT_FILTER, 2, 2);
    //---------------------------------------------------
    // 3) Block 3: 입력 pooled_x1 (24×16×20) -> 두 번의 Conv -> x2 (24×16×40)
    //---------------------------------------------------
    encode_block(
        pooled_x1, x2,
        batch_size, 
        UNET_PAR_ROW_NUM >> 2, UNET_PAR_COL_NUM >> 2,
        UNET_PAR_8_CONV2D_IN_FILTER,  UNET_PAR_8_CONV2D_OUT_FILTER,  UNET_PAR_8_CONV2D_KERNEL_Y, 1, 1,
        UNET_PAR_10_CONV2D_IN_FILTER, UNET_PAR_10_CONV2D_OUT_FILTER, UNET_PAR_10_CONV2D_KERNEL_Y, 1, 1,
        (float *)layer_8_conv2d_weight, (float *)layer_8_conv2d_bias,
        (float *)layer_9_batch_norm_gamma, (float *)layer_9_batch_norm_beta, 
        (float *)layer_9_batch_norm_mean,  (float *)layer_9_batch_norm_variance,
        (float *)layer_10_conv2d_weight, (float *)layer_10_conv2d_bias,
        (float *)layer_11_batch_norm_gamma, (float *)layer_11_batch_norm_beta, 
        (float *)layer_11_batch_norm_mean,  (float *)layer_11_batch_norm_variance
    );
    // x2의 shape: (24×16×40)

    // Block3 결과 x2에 max_pool -> pooled_x2 shape: (12×8×40)
    float ****pooled_x2 = alloc_4d_arr(
        batch_size,
        (UNET_PAR_ROW_NUM >> 2) >> 1,  // 24/2 = 12
        (UNET_PAR_COL_NUM >> 2) >> 1,  // 16/2 = 8
        UNET_PAR_8_CONV2D_OUT_FILTER   // 40
    );
    max_pool(x2, pooled_x2, batch_size, (UNET_PAR_ROW_NUM >> 2), (UNET_PAR_COL_NUM >> 2), UNET_PAR_8_CONV2D_OUT_FILTER, 2, 2);
    //---------------------------------------------------
    // 4) Block 4 (Bottleneck A): pooled_x2(12×8×40)-> x3(12×8×80)
    //---------------------------------------------------
    encode_block(
        pooled_x2, x3,
        batch_size, 
        UNET_PAR_ROW_NUM >> 3, UNET_PAR_COL_NUM >> 3,
        UNET_PAR_12_CONV2D_IN_FILTER, UNET_PAR_12_CONV2D_OUT_FILTER, UNET_PAR_12_CONV2D_KERNEL_Y, 1, 1,
        UNET_PAR_14_CONV2D_IN_FILTER, UNET_PAR_14_CONV2D_OUT_FILTER, UNET_PAR_14_CONV2D_KERNEL_Y, 1, 1,
        (float *)layer_12_conv2d_weight, (float *)layer_12_conv2d_bias,
        (float *)layer_13_batch_norm_gamma, (float *)layer_13_batch_norm_beta, 
        (float *)layer_13_batch_norm_mean,  (float *)layer_13_batch_norm_variance,
        (float *)layer_14_conv2d_weight, (float *)layer_14_conv2d_bias,
        (float *)layer_15_batch_norm_gamma, (float *)layer_15_batch_norm_beta, 
        (float *)layer_15_batch_norm_mean,  (float *)layer_15_batch_norm_variance
    );
    // x3의 shape: (12×8×80)
    // 여기서는 추가 max_pool 없이 그대로 x3 사용

    //---------------------------------------------------
    // 5) Block 5 (Bottleneck B): x3(12×8×80)-> x4(12×8×160)
    //---------------------------------------------------
    encode_block(
        x3, x4,
        batch_size, 
        UNET_PAR_ROW_NUM >> 3, UNET_PAR_COL_NUM >> 3,
        UNET_PAR_16_CONV2D_IN_FILTER, UNET_PAR_16_CONV2D_OUT_FILTER, UNET_PAR_16_CONV2D_KERNEL_Y, 1, 1,
        UNET_PAR_18_CONV2D_IN_FILTER, UNET_PAR_18_CONV2D_OUT_FILTER, UNET_PAR_18_CONV2D_KERNEL_Y, 1, 1,
        (float *)layer_16_conv2d_weight, (float *)layer_16_conv2d_bias,
        (float *)layer_17_batch_norm_gamma, (float *)layer_17_batch_norm_beta, 
        (float *)layer_17_batch_norm_mean,  (float *)layer_17_batch_norm_variance,
        (float *)layer_18_conv2d_weight, (float *)layer_18_conv2d_bias,
        (float *)layer_19_batch_norm_gamma, (float *)layer_19_batch_norm_beta, 
        (float *)layer_19_batch_norm_mean,  (float *)layer_19_batch_norm_variance
    );
    // x4의 shape: (12×8×160)

    //---------------------------------------------------
    // 메모리 해제 (pooled_x0, pooled_x1, pooled_x2)
    //---------------------------------------------------
    // pooled_x0 : shape (48×32×10)
    free_4d_arr(pooled_x0, batch_size, 
                UNET_PAR_ROW_NUM >> 1, 
                UNET_PAR_COL_NUM >> 1);

    // pooled_x1 : shape (24×16×20)
    free_4d_arr(pooled_x1, batch_size, 
                (UNET_PAR_ROW_NUM >> 1) >> 1,  // 48/2 = 24
                (UNET_PAR_COL_NUM >> 1) >> 1); // 32/2 = 16

    // pooled_x2 : shape (12×8×40)
    free_4d_arr(pooled_x2, batch_size, 
                (UNET_PAR_ROW_NUM >> 2) >> 1,  // 24/2 = 12
                (UNET_PAR_COL_NUM >> 2) >> 1); // 16/2 = 8

    return;
}



/*======================================================================
   디코딩 (decoding)
======================================================================*/
void decoding(
    float ****bottleneck,  // 인코더의 최종 x4: (12×8×160)
    float ****output,      // 최종 출력(82×67×1)
    // skip 연결
    float ****x0,  // (96×64×20)
    float ****x1,  // (48×32×40)
    float ****x2,  // (24×16×80)
    float ****x3,  // (12×8×80) - (쓰임새가 있다면; 없으면 제거)
    int batch_size
)
{
    //----------------------------------------------------------------------
    // 1) decode_block(bottleneck=12×8×160, skip=x2=24×16×80) => d0=24×16×80
    //----------------------------------------------------------------------
    float ****d0 = alloc_4d_arr(
        batch_size,
        /*24 = (12<<1)*/ (UNET_PAR_ROW_NUM >> 3) << 1,
        /*16 = (8<<1) */ (UNET_PAR_COL_NUM >> 3) << 1,
        /*80*/ UNET_PAR_22_CONV2D_OUT_FILTER
    );
    decode_block(
        bottleneck, x2, d0,
        batch_size,
        (UNET_PAR_ROW_NUM >> 3),  // 12
        (UNET_PAR_COL_NUM >> 3),  // 8
        // conv2d_transpose
        UNET_PAR_20_CONV2D_TRANSPOSE_IN_FILTER,  
        UNET_PAR_20_CONV2D_TRANSPOSE_OUT_FILTER, 
        UNET_PAR_20_CONV2D_TRANSPOSE_KERNEL_Y, 2, 1,
        // conv2d
        UNET_PAR_22_CONV2D_IN_FILTER,  
        UNET_PAR_22_CONV2D_OUT_FILTER, 
        UNET_PAR_22_CONV2D_KERNEL_Y, 1, 1,
        (float *)layer_20_conv2d_transpose_weight, 
        (float *)layer_20_conv2d_transpose_bias,
        (float *)layer_21_batch_norm_gamma, (float *)layer_21_batch_norm_beta, 
        (float *)layer_21_batch_norm_mean,  (float *)layer_21_batch_norm_variance,
        (float *)layer_22_conv2d_weight, (float *)layer_22_conv2d_bias,
        (float *)layer_23_batch_norm_gamma, (float *)layer_23_batch_norm_beta, 
        (float *)layer_23_batch_norm_mean,  (float *)layer_23_batch_norm_variance
    );
    // d0 shape: (24×16×80)

    //----------------------------------------------------------------------
    // 2) decode_block(d0=24×16×80, skip=x1=48×32×40) => d1=48×32×40
    //----------------------------------------------------------------------
    float ****d1 = alloc_4d_arr(
        batch_size,
        /*48 = (24<<1)*/ (UNET_PAR_ROW_NUM >> 2) << 1,
        /*32 = (16<<1)*/ (UNET_PAR_COL_NUM >> 2) << 1,
        /*40*/ UNET_PAR_26_CONV2D_OUT_FILTER
    );
    decode_block(
        d0, x1, d1,
        batch_size,
        (UNET_PAR_ROW_NUM >> 2), //24
        (UNET_PAR_COL_NUM >> 2), //16
        // conv2d_transpose
        UNET_PAR_24_CONV2D_TRANSPOSE_IN_FILTER,  
        UNET_PAR_24_CONV2D_TRANSPOSE_OUT_FILTER, 
        UNET_PAR_24_CONV2D_TRANSPOSE_KERNEL_Y, 2, 1,
        // conv2d
        UNET_PAR_26_CONV2D_IN_FILTER, 
        UNET_PAR_26_CONV2D_OUT_FILTER, 
        UNET_PAR_26_CONV2D_KERNEL_Y, 1, 1,
        (float *)layer_24_conv2d_transpose_weight, 
        (float *)layer_24_conv2d_transpose_bias,
        (float *)layer_25_batch_norm_gamma, (float *)layer_25_batch_norm_beta, 
        (float *)layer_25_batch_norm_mean,  (float *)layer_25_batch_norm_variance,
        (float *)layer_26_conv2d_weight, (float *)layer_26_conv2d_bias,
        (float *)layer_27_batch_norm_gamma, (float *)layer_27_batch_norm_beta, 
        (float *)layer_27_batch_norm_mean,  (float *)layer_27_batch_norm_variance
    );
    // d1 shape: (48×32×40)

    //----------------------------------------------------------------------
    // 3) decode_block(d1=48×32×40, skip=x0=96×64×20) => d2=96×64×20
    //----------------------------------------------------------------------
    float ****d2 = alloc_4d_arr(
        batch_size,
        (UNET_PAR_ROW_NUM >> 1) << 1, //48<<1=96
        (UNET_PAR_COL_NUM >> 1) << 1, //32<<1=64
        UNET_PAR_30_CONV2D_OUT_FILTER //20
    );
    decode_block(
        d1, x0, d2,
        batch_size,
        (UNET_PAR_ROW_NUM >> 1), //48
        (UNET_PAR_COL_NUM >> 1), //32
        // conv2d_transpose
        UNET_PAR_28_CONV2D_TRANSPOSE_IN_FILTER,  
        UNET_PAR_28_CONV2D_TRANSPOSE_OUT_FILTER, 
        UNET_PAR_28_CONV2D_TRANSPOSE_KERNEL_Y, 2, 1,
        // conv2d
        UNET_PAR_30_CONV2D_IN_FILTER, 
        UNET_PAR_30_CONV2D_OUT_FILTER, 
        UNET_PAR_30_CONV2D_KERNEL_Y, 1, 1,
        (float *)layer_28_conv2d_transpose_weight, 
        (float *)layer_28_conv2d_transpose_bias,
        (float *)layer_29_batch_norm_gamma, (float *)layer_29_batch_norm_beta, 
        (float *)layer_29_batch_norm_mean,  (float *)layer_29_batch_norm_variance,
        (float *)layer_30_conv2d_weight, (float *)layer_30_conv2d_bias,
        (float *)layer_31_batch_norm_gamma, (float *)layer_31_batch_norm_beta, 
        (float *)layer_31_batch_norm_mean,  (float *)layer_31_batch_norm_variance
    );
    // d2 shape: (96×64×20)

    //----------------------------------------------------------------------
    // 4) 최종 Conv2D(커널=1) => d3(96×64×1) -> resize(82×67×1)
    //----------------------------------------------------------------------
    float ****d3 = alloc_4d_arr(
        batch_size,
        UNET_PAR_ROW_NUM,     //96
        UNET_PAR_COL_NUM,     //64
        UNET_PAR_32_CONV2D_OUT_FILTER //1
    );
    conv2d(
        d2, d3,
        (float *)layer_32_conv2d_weight, (float *)layer_32_conv2d_bias,
        batch_size,
        UNET_PAR_32_CONV2D_IN_FILTER,  //20
        UNET_PAR_32_CONV2D_OUT_FILTER, //1
        UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM,
        UNET_PAR_32_CONV2D_KERNEL_Y,  //1
        1, 0  // stride=1, pad=0
    );
    // d3: (96×64×1)

    // 최종 resize => (82×67×1)
    resize_bilinear_batch(
        d3, output,
        batch_size,
        UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM,      
        CMAQ_PAR_ROW_NUM, CMAQ_PAR_COL_NUM,      
        UNET_PAR_32_CONV2D_OUT_FILTER            
    );

    // d0: (24×16×80)
    free_4d_arr(d0, batch_size, (UNET_PAR_ROW_NUM >> 3) << 1, (UNET_PAR_COL_NUM >> 3) << 1);
    // d1: (48×32×40)
    free_4d_arr(d1, batch_size, (UNET_PAR_ROW_NUM >> 2), (UNET_PAR_COL_NUM >> 2));
    // d2: (96×64×20)
    free_4d_arr(d2, batch_size, (UNET_PAR_ROW_NUM >> 1), (UNET_PAR_COL_NUM >> 1));
    // d3: (96×64×1)
    free_4d_arr(d3, batch_size, UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM);

    return;
}

/*======================================================================
   forward
======================================================================*/
void forward(float ***input, float ****output, int batch_size)
{
    float ****gridded_input = alloc_4d_arr(batch_size,
                                           CMAQ_PAR_ROW_NUM, CMAQ_PAR_COL_NUM,
                                           CMAQ_PAR_SECTOR_NUM);
    float ****resized_gridded_input = alloc_4d_arr(batch_size,
                                                   UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM,
                                                   CMAQ_PAR_SECTOR_NUM);
    gridding(input, gridded_input, batch_size,
             CMAQ_PAR_REGION_NUM, CMAQ_PAR_SECTOR_NUM);
    resize_bilinear_batch(gridded_input, resized_gridded_input, batch_size,
                          CMAQ_PAR_ROW_NUM, CMAQ_PAR_COL_NUM,
                          UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM,
                          CMAQ_PAR_SECTOR_NUM);
    
    // 인코딩 과정:
    // Block 1: (96,64,2) → x0: (96,64,10)
    float ****x0 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM, UNET_PAR_2_CONV2D_OUT_FILTER);
    // Block 2: (96,64,10) → conv 연산 → x1: (96,64,20) → max_pool → (48,32,20)
    float ****x1 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM >> 1, UNET_PAR_COL_NUM >> 1, UNET_PAR_6_CONV2D_OUT_FILTER);
    // Block 3: (max_pool된 결과) → x2: (48,32,40) → max_pool → (24,16,40)
    float ****x2 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM >> 2, UNET_PAR_COL_NUM >> 2, UNET_PAR_10_CONV2D_OUT_FILTER);
    // Block 4: (max_pool된 결과) → x3: (24,16,80) → max_pool → (12,8,80)
    float ****x3 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM >> 3, UNET_PAR_COL_NUM >> 3, UNET_PAR_12_CONV2D_OUT_FILTER);
    // Block 5: (max_pool 미적용) → x4: (12,8,160)
    float ****x4 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM >> 3, UNET_PAR_COL_NUM >> 3, UNET_PAR_16_CONV2D_OUT_FILTER);
    
    encoding(resized_gridded_input, x0, x1, x2, x3, x4, batch_size);

    decoding(x4, output, x0, x1, x2, x3, batch_size);

    free_4d_arr(gridded_input, batch_size, CMAQ_PAR_ROW_NUM, CMAQ_PAR_COL_NUM);
    free_4d_arr(resized_gridded_input, batch_size, UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM);
    free_4d_arr(x0, batch_size, UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM);
    free_4d_arr(x1, batch_size, UNET_PAR_ROW_NUM >> 1, UNET_PAR_COL_NUM >> 1);
    free_4d_arr(x2, batch_size, UNET_PAR_ROW_NUM >> 2, UNET_PAR_COL_NUM >> 2);
    free_4d_arr(x3, batch_size, UNET_PAR_ROW_NUM >> 3, UNET_PAR_COL_NUM >> 3);
    free_4d_arr(x4, batch_size, UNET_PAR_ROW_NUM >> 3, UNET_PAR_COL_NUM >> 3);
    return;
}

/*======================================================================
   CALL API 및 SET FUNCTIONS
======================================================================*/
void call(float *c_inputs, float *c_outputs, int batch_size)
{
    float ***inputs = alloc_3d_arr(batch_size, CMAQ_PAR_REGION_NUM, CMAQ_PAR_SECTOR_NUM);
    float ****outputs = alloc_4d_arr(batch_size, CMAQ_PAR_ROW_NUM, CMAQ_PAR_COL_NUM, UNET_PAR_32_CONV2D_OUT_FILTER);
    // 1차원 입력 배열 c_inputs를 3차원 배열 inputs로 복사
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < CMAQ_PAR_REGION_NUM; j++) {
            for (size_t k = 0; k < CMAQ_PAR_SECTOR_NUM; k++) {
                inputs[i][j][k] = c_inputs[i * CMAQ_PAR_REGION_NUM * CMAQ_PAR_SECTOR_NUM +
                                            j * CMAQ_PAR_SECTOR_NUM + k];
            }
        }
    }
    forward(inputs, outputs, batch_size);
    // 3차원 outputs를 1차원 배열 c_outputs로 복사
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < CMAQ_PAR_ROW_NUM; j++) {
            for (size_t k = 0; k < CMAQ_PAR_COL_NUM; k++) {
                for (size_t l = 0; l < UNET_PAR_32_CONV2D_OUT_FILTER; l++) {
                    c_outputs[i * CMAQ_PAR_ROW_NUM * CMAQ_PAR_COL_NUM * UNET_PAR_32_CONV2D_OUT_FILTER +
                              j * CMAQ_PAR_COL_NUM * UNET_PAR_32_CONV2D_OUT_FILTER +
                              k * UNET_PAR_32_CONV2D_OUT_FILTER + l] = outputs[i][j][k][l];
                }
            }
        }
    }
    free_3d_arr(inputs, batch_size, CMAQ_PAR_REGION_NUM);
    free_4d_arr(outputs, batch_size, CMAQ_PAR_ROW_NUM, CMAQ_PAR_COL_NUM);
    return;
}

float *set_sector(void)
{
    int status;
    float *sector = (float *)malloc(CMAQ_PAR_SECTOR_NUM * sizeof(float));
    status = scanf("%f %f", &sector[0], &sector[1]);
    return sector;
}

float ***set_ctrl_mat(int batch_size)
{
    const char* region_codes = "ABCDEFGHIJKLMNOPQ";
    float ***ctrl_mat = (float ***)malloc(batch_size * sizeof(float **));
    for (size_t b = 0; b < batch_size; b++) {
        ctrl_mat[b] = (float **)malloc(CMAQ_PAR_REGION_NUM * sizeof(float *));
        for (size_t i = 0; i < CMAQ_PAR_REGION_NUM; i++) {
            printf("Enter the %c sector values: ", region_codes[i]);
            ctrl_mat[b][i] = (float *)set_sector();
        }
    }
    return ctrl_mat;
}

int main(void)
{
    int batch_size = 1;
    float ***inputs = set_ctrl_mat(batch_size);
    float ****outputs = alloc_4d_arr(batch_size, CMAQ_PAR_ROW_NUM, CMAQ_PAR_COL_NUM, 1);

    forward(inputs, outputs, batch_size);

    free_3d_arr(inputs, batch_size, CMAQ_PAR_REGION_NUM);
    free_4d_arr(outputs, batch_size, CMAQ_PAR_ROW_NUM, CMAQ_PAR_COL_NUM);
    return 0;
}
