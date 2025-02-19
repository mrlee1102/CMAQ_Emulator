#include "forward.h"

/*----------------------------------------------------------
 * 1) 인코더 블록: encode_block
 *    (Conv2D → BN → Activation → Conv2D → BN → Activation → MaxPool)
 *----------------------------------------------------------*/
void encode_block(
    float ****input, float ****output_0, float ****output_1,
    int batch_size, int height, int width,
    // 첫 번째 conv2d의 파라미터
    int filter_0, int filter_1, int kernel_0, int stride_0, int padding_0,
    // 두 번째 conv2d의 파라미터
    int filter_2, int filter_3, int kernel_1, int stride_1, int padding_1,
    // 세 번째 conv2d의 파라미터
    int filter_4, int filter_5, int kernel_2, int stride_2, int padding_2,
    // 첫 번째 conv2d의 가중치/편향
    const float *weight_0, const float *bias_0,
    // 두 번째 conv2d의 가중치/편향
    const float *weight_1, const float *bias_1,
    // 첫 번째 BN (두 번째 conv2d 후)의 파라미터
    const float *gamma_0, const float *beta_0, const float *mean_0, const float *var_0,
    // 세 번째 conv2d의 가중치/편향
    const float *weight_2, const float *bias_2,
    // 두 번째 BN (세 번째 conv2d 후)의 파라미터
    const float *gamma_1, const float *beta_1, const float *mean_1, const float *var_1
) 
{
    // 임시 버퍼: 첫 번째 conv2d 결과 저장 (크기: [batch, height, width, filter_1])
    float ****tmp_0 = alloc_4d_arr(batch_size, height, width, filter_1);

    // [1] 첫 번째 Conv2D: input → tmp_0
    conv2d(input, tmp_0, weight_0, bias_0, batch_size,
           filter_0, filter_1, height, width, kernel_0, stride_0, padding_0);

    // [2] 두 번째 Conv2D: tmp_0 → tmp_0 (채널 수 변경: filter_2→filter_3)
    conv2d(tmp_0, tmp_0, weight_1, bias_1, batch_size,
           filter_2, filter_3, height, width, kernel_1, stride_1, padding_1);

    // [3] 첫 번째 Batch Normalization 및 SILU 활성화 (두 번째 conv2d의 출력에 대해)
    batch_normalization(tmp_0, gamma_0, beta_0, mean_0, var_0,
                        batch_size, height, width, filter_3, 0.001, 0.99);
    silu(tmp_0, batch_size, height, width, filter_3);

    // [4] 세 번째 Conv2D: tmp_0 → output_1
    conv2d(tmp_0, output_1, weight_2, bias_2, batch_size,
           filter_4, filter_5, height, width, kernel_2, stride_2, padding_2);

    // [5] 두 번째 Batch Normalization 및 SILU 활성화 (세 번째 conv2d의 출력에 대해)
    batch_normalization(output_1, gamma_1, beta_1, mean_1, var_1,
                        batch_size, height, width, filter_5, 0.001, 0.99);
    silu(output_1, batch_size, height, width, filter_5);

    // [6] Max Pooling: output_1 → output_0 (채널 수: filter_5)
    max_pool(output_1, output_0, batch_size, height, width, filter_5, 2, 2);

    free_4d_arr(tmp_0, batch_size, height, width);
}

/*----------------------------------------------------------
 * 2) 디코더 블록: decode_block
 *    (Conv2DTranspose → BN → Activation → Concat → Conv2D → Conv2D → BN → Activation)
 *----------------------------------------------------------*/
void decode_block(
    float ****input_0, float ****input_1, float ****output,
    int batch_size, int height, int width,
    // [A] Conv2DTranspose 관련 파라미터
    int filter_0, int filter_1, int kernel_0, int stride_0, int padding_0,
    // [B] 첫 번째 Conv2D (Concat 후) 관련 파라미터
    int filter_2, int filter_3, int kernel_1, int stride_1, int padding_1,
    // [C] 두 번째 Conv2D 관련 파라미터
    int filter_4, int filter_5, int kernel_2, int stride_2, int padding_2,
    // [D] Conv2DTranspose 가중치/편향
    const float *weight_0, const float *bias_0,
    // [E] 첫 번째 Batch Normalization (Conv2DTranspose 후) 파라미터
    const float *gamma_0, const float *beta_0, const float *mean_0, const float *var_0,
    // [F] 첫 번째 Conv2D 가중치/편향 (Concat 후)
    const float *weight_1, const float *bias_1,
    // [H] 두 번째 Conv2D 가중치/편향
    const float *weight_2, const float *bias_2,
    // [G] 두 번째 Batch Normalization (첫 번째 Conv2D 후) 파라미터
    const float *gamma_1, const float *beta_1, const float *mean_1, const float *var_1
)
{
    // [1] 임시 버퍼 할당: Conv2DTranspose의 결과 저장
    //     크기: [batch_size, height*2, width*2, filter_1]
    float ****tmp_0 = alloc_4d_arr(batch_size, height << 1, width << 1, filter_1);
    
    // [2] 임시 버퍼 할당: Concat 결과 저장
    //     (두 입력(tmp_0와 input_1)의 채널 수가 같다고 가정하면, 최종 채널 수는 2*filter_1이어야 함.
    //      이 값은 filter_2로 정의되어 있다고 가정)
    float ****tmp_1 = alloc_4d_arr(batch_size, height << 1, width << 1, filter_2);
    
    // [3] 임시 버퍼 할당: 첫 번째 Conv2D 결과 저장
    //     크기: [batch_size, height*2, width*2, filter_3]
    float ****tmp_2 = alloc_4d_arr(batch_size, height << 1, width << 1, filter_3);
    
    // [A] Conv2DTranspose: input_0 → tmp_0
    conv2d_transpose(
        input_0, tmp_0,
        weight_0, bias_0,
        batch_size,
        filter_0, filter_1,
        height, width,
        kernel_0, stride_0, padding_0
    );
    
    // [B] 첫 번째 Batch Normalization + SILU (Conv2DTranspose 후)
    batch_normalization(
        tmp_0,
        gamma_0, beta_0, mean_0, var_0,
        batch_size, (height << 1), (width << 1), filter_1,
        0.001, 0.99
    );
    silu(tmp_0, batch_size, (height << 1), (width << 1), filter_1);
    
    // [C] Concat: tmp_0와 input_1를 채널 방향으로 결합하여 tmp_1에 저장
    concatenate(
        tmp_0,        // 업샘플링된 feature map
        input_1,      // skip-connection feature map
        tmp_1,        // 결합 결과
        batch_size,
        (height << 1), (width << 1),
        filter_1,     // tmp_0의 채널 수
        filter_1      // input_1의 채널 수 (동일하다고 가정)
    );
    
    // [D] 첫 번째 Conv2D: Concat 결과(tmp_1)를 입력으로 하여 feature map 생성 → tmp_2
    conv2d(
        tmp_1, tmp_2,
        weight_1, bias_1,
        batch_size,
        filter_2, filter_3,
        (height << 1), (width << 1),
        kernel_1, stride_1, padding_1
    );

    conv2d(
        tmp_2, output,
        weight_2, bias_2,
        batch_size,
        filter_4, filter_5,
        (height << 1), (width << 1),
        kernel_2, stride_2, padding_2
    );
    
    // [E] 두 번째 Batch Normalization + SILU (첫 번째 Conv2D 후)
    batch_normalization(
        output,
        gamma_1, beta_1, mean_1, var_1,
        batch_size, (height << 1), (width << 1), filter_5,
        0.001, 0.99
    );
    silu(output, batch_size, (height << 1), (width << 1), filter_5);
    
    // 임시 버퍼 해제
    free_4d_arr(tmp_0, batch_size, height << 1, width << 1);
    free_4d_arr(tmp_1, batch_size, height << 1, width << 1);
    free_4d_arr(tmp_2, batch_size, height << 1, width << 1);
}



/*----------------------------------------------------------
 * 3) 인코딩: 여러 encode_block 호출을 통해 skip connection 값(x0~x4) 생성
 *----------------------------------------------------------*/
void encoding(float ****input, float ****x0, float ****x1, float ****x2, float ****x3, float ****x4, int batch_size) 
{
    // 임시 버퍼 할당 (각 블록 사이의 출력 저장)
    float ****tmp0 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM >> 1, UNET_PAR_COL_NUM >> 1, UNET_PAR_3_CONV2D_OUT_FILTER);
    float ****tmp1 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM >> 2, UNET_PAR_COL_NUM >> 2, UNET_PAR_8_CONV2D_OUT_FILTER);
    float ****tmp2 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM >> 3, UNET_PAR_COL_NUM >> 3, UNET_PAR_13_CONV2D_OUT_FILTER);
    float ****tmp3 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM >> 4, UNET_PAR_COL_NUM >> 4, UNET_PAR_18_CONV2D_OUT_FILTER);
    float ****tmp4 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM >> 5, UNET_PAR_COL_NUM >> 5, UNET_PAR_23_CONV2D_OUT_FILTER);

    // 인코더 블록 #1
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

    // 인코더 블록 #2
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

    // 인코더 블록 #3
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

    // 인코더 블록 #4
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

    // 인코더 블록 #5
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

    // 임시 버퍼 해제
    free_4d_arr(tmp0, batch_size, UNET_PAR_ROW_NUM >> 1, UNET_PAR_COL_NUM >> 1);
    free_4d_arr(tmp1, batch_size, UNET_PAR_ROW_NUM >> 2, UNET_PAR_COL_NUM >> 2);
    free_4d_arr(tmp2, batch_size, UNET_PAR_ROW_NUM >> 3, UNET_PAR_COL_NUM >> 3);
    free_4d_arr(tmp3, batch_size, UNET_PAR_ROW_NUM >> 4, UNET_PAR_COL_NUM >> 4);
    free_4d_arr(tmp4, batch_size, UNET_PAR_ROW_NUM >> 5, UNET_PAR_COL_NUM >> 5);
    return;
}

/*----------------------------------------------------------
 * 4) 디코딩: 여러 번 decode_block 호출 및 최종 Conv2D + Resize
 *----------------------------------------------------------*/
void decoding(float ****input, float ****output, float ****x0, float ****x1, float ****x2, float ****x3, int batch_size) 
{
    // 임시 버퍼 할당
    float ****tmp0 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM >> 3, UNET_PAR_COL_NUM >> 3, UNET_PAR_28_CONV2D_OUT_FILTER);
    float ****tmp1 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM >> 2, UNET_PAR_COL_NUM >> 2, UNET_PAR_33_CONV2D_OUT_FILTER);
    float ****tmp2 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM >> 1, UNET_PAR_COL_NUM >> 1, UNET_PAR_38_CONV2D_OUT_FILTER);
    float ****tmp3 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM, UNET_PAR_43_CONV2D_OUT_FILTER);
    float ****tmp4 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM, UNET_PAR_45_CONV2D_OUT_FILTER); 

    // 디코더 블록 #1
    decode_block(
        input, x3, tmp0,
        batch_size, UNET_PAR_ROW_NUM >> 4, UNET_PAR_COL_NUM >> 4,
        UNET_PAR_25_CONV2D_TRANSPOSE_IN_FILTER, UNET_PAR_25_CONV2D_TRANSPOSE_OUT_FILTER, UNET_PAR_25_CONV2D_TRANSPOSE_KERNEL_Y, 2, 0,
        UNET_PAR_27_CONV2D_IN_FILTER, UNET_PAR_27_CONV2D_OUT_FILTER, UNET_PAR_27_CONV2D_KERNEL_Y, 1, 1,
        UNET_PAR_28_CONV2D_IN_FILTER, UNET_PAR_28_CONV2D_OUT_FILTER, UNET_PAR_28_CONV2D_KERNEL_Y, 1, 1,
        (float *)layer_25_conv2d_transpose_weight, (float *)layer_25_conv2d_transpose_bias,
        (float *)layer_26_batch_normalization_gamma, (float *)layer_26_batch_normalization_beta, (float *)layer_26_batch_normalization_mean, (float *)layer_26_batch_normalization_variance,
        (float *)layer_27_conv2d_weight, (float *)layer_27_conv2d_bias,
        (float *)layer_28_conv2d_weight, (float *)layer_28_conv2d_bias,
        (float *)layer_29_batch_normalization_gamma, (float *)layer_29_batch_normalization_beta, (float *)layer_29_batch_normalization_mean, (float *)layer_29_batch_normalization_variance
    );

    // 디코더 블록 #2
    decode_block(
        tmp0, x2, tmp1,
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

    // 디코더 블록 #3
    decode_block(
        tmp1, x1, tmp2,
        batch_size, UNET_PAR_ROW_NUM >> 2, UNET_PAR_COL_NUM >> 2,
        UNET_PAR_35_CONV2D_TRANSPOSE_IN_FILTER, UNET_PAR_35_CONV2D_TRANSPOSE_OUT_FILTER, UNET_PAR_35_CONV2D_TRANSPOSE_KERNEL_Y, 2, 0,
        UNET_PAR_37_CONV2D_IN_FILTER, UNET_PAR_37_CONV2D_OUT_FILTER, UNET_PAR_37_CONV2D_KERNEL_Y, 1, 1,
        UNET_PAR_38_CONV2D_IN_FILTER, UNET_PAR_38_CONV2D_OUT_FILTER, UNET_PAR_38_CONV2D_KERNEL_Y, 1, 1,
        (float *)layer_35_conv2d_transpose_weight, (float *)layer_35_conv2d_transpose_bias,
        (float *)layer_36_batch_normalization_gamma, (float *)layer_36_batch_normalization_beta, (float *)layer_36_batch_normalization_mean, (float *)layer_36_batch_normalization_variance,
        (float *)layer_37_conv2d_weight, (float *)layer_37_conv2d_bias,
        (float *)layer_38_conv2d_weight, (float *)layer_38_conv2d_bias,
        (float *)layer_39_batch_normalization_gamma, (float *)layer_39_batch_normalization_beta, (float *)layer_39_batch_normalization_mean, (float *)layer_39_batch_normalization_variance
    );

    // 디코더 블록 #4
    decode_block(
        tmp2, x0, tmp3,
        batch_size, UNET_PAR_ROW_NUM >> 1, UNET_PAR_COL_NUM >> 1,
        UNET_PAR_40_CONV2D_TRANSPOSE_IN_FILTER, UNET_PAR_40_CONV2D_TRANSPOSE_OUT_FILTER, UNET_PAR_40_CONV2D_TRANSPOSE_KERNEL_Y, 2, 0,
        UNET_PAR_42_CONV2D_IN_FILTER, UNET_PAR_42_CONV2D_OUT_FILTER, UNET_PAR_42_CONV2D_KERNEL_Y, 1, 1,
        UNET_PAR_43_CONV2D_IN_FILTER, UNET_PAR_43_CONV2D_OUT_FILTER, UNET_PAR_43_CONV2D_KERNEL_Y, 1, 1,
        (float *)layer_40_conv2d_transpose_weight, (float *)layer_40_conv2d_transpose_bias,
        (float *)layer_41_batch_normalization_gamma, (float *)layer_41_batch_normalization_beta, (float *)layer_41_batch_normalization_mean, (float *)layer_41_batch_normalization_variance,
        (float *)layer_42_conv2d_weight, (float *)layer_42_conv2d_bias,
        (float *)layer_43_conv2d_weight, (float *)layer_43_conv2d_bias,
        (float *)layer_44_batch_normalization_gamma, (float *)layer_44_batch_normalization_beta, (float *)layer_44_batch_normalization_mean, (float *)layer_44_batch_normalization_variance
    );

    // 마지막 Conv2D (최종 출력 생성)
    conv2d(
        tmp3, tmp4, (float *)layer_45_conv2d_weight, (float *)layer_45_conv2d_bias,
        batch_size, UNET_PAR_45_CONV2D_IN_FILTER, UNET_PAR_45_CONV2D_OUT_FILTER,
        UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM, UNET_PAR_45_CONV2D_KERNEL_Y, 1, 0
    );
    // 필요 시 최종 결과를 원본 그리드 크기로 resize
    resize_bilinear_batch(
        tmp4, output, batch_size,
        UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM,
        CMAQ_PAR_ROW_NUM, CMAQ_PAR_COL_NUM,
        UNET_PAR_45_CONV2D_OUT_FILTER  // 채널 수에 해당하는 매크로
    );

    // 임시 버퍼 해제
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
    // 1) Gridding 및 Resizing
    float ****gridded_input = alloc_4d_arr(batch_size, CMAQ_PAR_ROW_NUM, CMAQ_PAR_COL_NUM, CMAQ_PAR_SECTOR_NUM);
    float ****resized_gridded_input = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM, CMAQ_PAR_SECTOR_NUM);
    gridding(input, gridded_input, batch_size, CMAQ_PAR_REGION_NUM, CMAQ_PAR_SECTOR_NUM);
    resize_bilinear_batch(gridded_input, resized_gridded_input, batch_size,
                          CMAQ_PAR_ROW_NUM, CMAQ_PAR_COL_NUM,
                          UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM,
                          CMAQ_PAR_SECTOR_NUM);

    // 2) 인코딩: skip connection 값 x0~x4 생성
    float ****x0 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM, UNET_PAR_3_CONV2D_OUT_FILTER);
    float ****x1 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM >> 1, UNET_PAR_COL_NUM >> 1, UNET_PAR_8_CONV2D_OUT_FILTER);
    float ****x2 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM >> 2, UNET_PAR_COL_NUM >> 2, UNET_PAR_13_CONV2D_OUT_FILTER);
    float ****x3 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM >> 3, UNET_PAR_COL_NUM >> 3, UNET_PAR_18_CONV2D_OUT_FILTER);
    float ****x4 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM >> 4, UNET_PAR_COL_NUM >> 4, UNET_PAR_23_CONV2D_OUT_FILTER);
    encoding(resized_gridded_input, x0, x1, x2, x3, x4, batch_size);

    // 3) 디코딩: 인코딩 결과와 skip connection을 활용하여 최종 출력 생성
    decoding(x4, output, x0, x1, x2, x3, batch_size);

    // 4) 임시 버퍼 해제
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

    // 1D 입력 배열을 3D 배열로 매핑
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < CMAQ_PAR_REGION_NUM; j++) {
            for (size_t k = 0; k < CMAQ_PAR_SECTOR_NUM; k++) {
                inputs[i][j][k] = c_inputs[i * CMAQ_PAR_REGION_NUM * CMAQ_PAR_SECTOR_NUM + j * CMAQ_PAR_SECTOR_NUM + k];
            }
        }
    }

    forward(inputs, outputs, batch_size);

    // 3D 출력 배열을 1D 배열로 매핑
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

    forward(inputs, outputs, batch_size);

    free_3d_arr(inputs, batch_size, CMAQ_PAR_REGION_NUM);
    free_4d_arr(outputs, batch_size, CMAQ_PAR_ROW_NUM, CMAQ_PAR_COL_NUM);
    return 0;
}
