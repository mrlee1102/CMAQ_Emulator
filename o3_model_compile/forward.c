#include "forward.h"
#include <math.h>

void encode_block(
    float ****input,
    float ****output_skip,
    int batch_size, int height, int width,
    int filter_0, int filter_1, int kernel_0, int stride_0, int padding_0,
    int filter_2, int filter_3, int kernel_1, int stride_1, int padding_1,
    const float *weight_0, const float *bias_0,
    const float *gamma_0, const float *beta_0, const float *mean_0, const float *var_0,
    const float *weight_1, const float *bias_1,
    const float *gamma_1, const float *beta_1, const float *mean_1, const float *var_1
) {
    float ****tmp = alloc_4d_arr(batch_size, height, width, filter_1);
    conv2d(input, tmp, weight_0, bias_0, batch_size, filter_0, filter_1, height, width, kernel_0, stride_0, padding_0);
    batch_normalization(tmp, gamma_0, beta_0, mean_0, var_0, batch_size, height, width, filter_1, 0.001, 0.99);
    silu(tmp, batch_size, height, width, filter_1);
    conv2d(tmp, output_skip, weight_1, bias_1, batch_size, filter_2, filter_3, height, width, kernel_1, stride_1, padding_1);
    batch_normalization(output_skip, gamma_1, beta_1, mean_1, var_1, batch_size, height, width, filter_3, 0.001, 0.99);
    silu(output_skip, batch_size, height, width, filter_3);
    free_4d_arr(tmp, batch_size, height, width);
    return;
}

void decode_block(
    float ****input_0, float ****input_1, float ****output,
    int batch_size, int height, int width,
    int filter_0, int filter_1, int kernel_0, int stride_0, int padding_0,
    int filter_2, int filter_3, int kernel_1, int stride_1, int padding_1,
    const float *weight_0, const float *bias_0,
    const float *gamma_0, const float *beta_0, const float *mean_0, const float *var_0,
    const float *weight_1, const float *bias_1,
    const float *gamma_1, const float *beta_1, const float *mean_1, const float *var_1
) {
    float ****tmp = alloc_4d_arr(batch_size, height << 1, width << 1, filter_1);
    float ****tmp2 = alloc_4d_arr(batch_size, height << 1, width << 1, filter_2);
    conv2d_transpose(input_0, tmp, weight_0, bias_0, batch_size, filter_0, filter_1, height, width, kernel_0, stride_0, padding_0);
    batch_normalization(tmp, gamma_0, beta_0, mean_0, var_0, batch_size, (height << 1), (width << 1), filter_1, 0.001, 0.99);
    silu(tmp, batch_size, (height << 1), (width << 1), filter_1);
    concatenate(tmp, input_1, tmp2, batch_size, (height << 1), (width << 1), filter_1, filter_1);
    conv2d(tmp2, output, weight_1, bias_1, batch_size, filter_2, filter_3, (height << 1), (width << 1), kernel_1, stride_1, padding_1);
    batch_normalization(output, gamma_1, beta_1, mean_1, var_1, batch_size, (height << 1), (width << 1), filter_3, 0.001, 0.99);
    silu(output, batch_size, (height << 1), (width << 1), filter_3);
    free_4d_arr(tmp, batch_size, (height << 1), (width << 1));
    free_4d_arr(tmp2, batch_size, (height << 1), (width << 1));
    return;
}

void encoding(
    float ****input,
    float ****x0, float ****x1, float ****x2, float ****x3, float ****x4,
    int batch_size
) {
    // Block 1: In: (96,64,2) -> Out: (96,64,10)
    encode_block(input, x0, batch_size, UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM,
        UNET_PAR_0_CONV2D_IN_FILTER, UNET_PAR_0_CONV2D_OUT_FILTER, UNET_PAR_0_CONV2D_KERNEL_Y, 1, 1,
        UNET_PAR_2_CONV2D_IN_FILTER, UNET_PAR_2_CONV2D_OUT_FILTER, UNET_PAR_2_CONV2D_KERNEL_Y, 1, 1,
        (float *)layer_0_conv2d_weight, (float *)layer_0_conv2d_bias,
        (float *)layer_1_batch_norm_gamma, (float *)layer_1_batch_norm_beta,
        (float *)layer_1_batch_norm_mean, (float *)layer_1_batch_norm_variance,
        (float *)layer_2_conv2d_weight, (float *)layer_2_conv2d_bias,
        (float *)layer_3_batch_norm_gamma, (float *)layer_3_batch_norm_beta,
        (float *)layer_3_batch_norm_mean, (float *)layer_3_batch_norm_variance
    );
    // MaxPool: (96,64,10) -> (48,32,10)
    float ****pooled_x0 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM >> 1, UNET_PAR_COL_NUM >> 1, UNET_PAR_2_CONV2D_OUT_FILTER);
    max_pool(x0, pooled_x0, batch_size, UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM, UNET_PAR_2_CONV2D_OUT_FILTER, 2, 2);
    // Block 2: In: (48,32,10) -> Out: (48,32,20)
    encode_block(pooled_x0, x1, batch_size, UNET_PAR_ROW_NUM >> 1, UNET_PAR_COL_NUM >> 1,
        UNET_PAR_4_CONV2D_IN_FILTER, UNET_PAR_4_CONV2D_OUT_FILTER, UNET_PAR_4_CONV2D_KERNEL_Y, 1, 1,
        UNET_PAR_6_CONV2D_IN_FILTER, UNET_PAR_6_CONV2D_OUT_FILTER, UNET_PAR_6_CONV2D_KERNEL_Y, 1, 1,
        (float *)layer_4_conv2d_weight, (float *)layer_4_conv2d_bias,
        (float *)layer_5_batch_norm_gamma, (float *)layer_5_batch_norm_beta,
        (float *)layer_5_batch_norm_mean, (float *)layer_5_batch_norm_variance,
        (float *)layer_6_conv2d_weight, (float *)layer_6_conv2d_bias,
        (float *)layer_7_batch_norm_gamma, (float *)layer_7_batch_norm_beta,
        (float *)layer_7_batch_norm_mean, (float *)layer_7_batch_norm_variance
    );
    // MaxPool: (48,32,20) -> (24,16,20)
    float ****pooled_x1 = alloc_4d_arr(batch_size, (UNET_PAR_ROW_NUM >> 1) >> 1, (UNET_PAR_COL_NUM >> 1) >> 1, UNET_PAR_4_CONV2D_OUT_FILTER);
    max_pool(x1, pooled_x1, batch_size, UNET_PAR_ROW_NUM >> 1, UNET_PAR_COL_NUM >> 1, UNET_PAR_4_CONV2D_OUT_FILTER, 2, 2);
    free_4d_arr(pooled_x0, batch_size, UNET_PAR_ROW_NUM >> 1, UNET_PAR_COL_NUM >> 1);
    // Block 3: In: (24,16,20) -> Out: (24,16,40)
    encode_block(pooled_x1, x2, batch_size, UNET_PAR_ROW_NUM >> 2, UNET_PAR_COL_NUM >> 2,
        UNET_PAR_8_CONV2D_IN_FILTER, UNET_PAR_8_CONV2D_OUT_FILTER, UNET_PAR_8_CONV2D_KERNEL_Y, 1, 1,
        UNET_PAR_10_CONV2D_IN_FILTER, UNET_PAR_10_CONV2D_OUT_FILTER, UNET_PAR_10_CONV2D_KERNEL_Y, 1, 1,
        (float *)layer_8_conv2d_weight, (float *)layer_8_conv2d_bias,
        (float *)layer_9_batch_norm_gamma, (float *)layer_9_batch_norm_beta,
        (float *)layer_9_batch_norm_mean, (float *)layer_9_batch_norm_variance,
        (float *)layer_10_conv2d_weight, (float *)layer_10_conv2d_bias,
        (float *)layer_11_batch_norm_gamma, (float *)layer_11_batch_norm_beta,
        (float *)layer_11_batch_norm_mean, (float *)layer_11_batch_norm_variance
    );
    // MaxPool: (24,16,40) -> (12,8,40)
    float ****pooled_x2 = alloc_4d_arr(batch_size, (UNET_PAR_ROW_NUM >> 2) >> 1, (UNET_PAR_COL_NUM >> 2) >> 1, UNET_PAR_8_CONV2D_OUT_FILTER);
    max_pool(x2, pooled_x2, batch_size, UNET_PAR_ROW_NUM >> 2, UNET_PAR_COL_NUM >> 2, UNET_PAR_8_CONV2D_OUT_FILTER, 2, 2);
    free_4d_arr(pooled_x1, batch_size, (UNET_PAR_ROW_NUM >> 1) >> 1, (UNET_PAR_COL_NUM >> 1) >> 1);
    // Block 4 (Bottleneck A): In: (12,8,40) -> Out: (12,8,80)
    encode_block(pooled_x2, x3, batch_size, UNET_PAR_ROW_NUM >> 3, UNET_PAR_COL_NUM >> 3,
        UNET_PAR_12_CONV2D_IN_FILTER, UNET_PAR_12_CONV2D_OUT_FILTER, UNET_PAR_12_CONV2D_KERNEL_Y, 1, 1,
        UNET_PAR_14_CONV2D_IN_FILTER, UNET_PAR_14_CONV2D_OUT_FILTER, UNET_PAR_14_CONV2D_KERNEL_Y, 1, 1,
        (float *)layer_12_conv2d_weight, (float *)layer_12_conv2d_bias,
        (float *)layer_13_batch_norm_gamma, (float *)layer_13_batch_norm_beta,
        (float *)layer_13_batch_norm_mean, (float *)layer_13_batch_norm_variance,
        (float *)layer_14_conv2d_weight, (float *)layer_14_conv2d_bias,
        (float *)layer_15_batch_norm_gamma, (float *)layer_15_batch_norm_beta,
        (float *)layer_15_batch_norm_mean, (float *)layer_15_batch_norm_variance
    );
    // Block 5 (Bottleneck B): In: (12,8,80) -> Out: (12,8,160)
    encode_block(x3, x4, batch_size, UNET_PAR_ROW_NUM >> 3, UNET_PAR_COL_NUM >> 3,
        UNET_PAR_16_CONV2D_IN_FILTER, UNET_PAR_16_CONV2D_OUT_FILTER, UNET_PAR_16_CONV2D_KERNEL_Y, 1, 1,
        UNET_PAR_18_CONV2D_IN_FILTER, UNET_PAR_18_CONV2D_OUT_FILTER, UNET_PAR_18_CONV2D_KERNEL_Y, 1, 1,
        (float *)layer_16_conv2d_weight, (float *)layer_16_conv2d_bias,
        (float *)layer_17_batch_norm_gamma, (float *)layer_17_batch_norm_beta,
        (float *)layer_17_batch_norm_mean, (float *)layer_17_batch_norm_variance,
        (float *)layer_18_conv2d_weight, (float *)layer_18_conv2d_bias,
        (float *)layer_19_batch_norm_gamma, (float *)layer_19_batch_norm_beta,
        (float *)layer_19_batch_norm_mean, (float *)layer_19_batch_norm_variance
    );
    free_4d_arr(pooled_x2, batch_size, (UNET_PAR_ROW_NUM >> 2) >> 1, (UNET_PAR_COL_NUM >> 2) >> 1);
}

void decoding(
    float ****bottleneck,
    float ****output,
    float ****x0,
    float ****x1,
    float ****x2,
    float ****x3,
    int batch_size
)
{
    float ****d0 = alloc_4d_arr(batch_size, (UNET_PAR_ROW_NUM >> 3) << 1, (UNET_PAR_COL_NUM >> 3) << 1, UNET_PAR_22_CONV2D_OUT_FILTER);
    decode_block(bottleneck, x2, d0, batch_size, (UNET_PAR_ROW_NUM >> 3), (UNET_PAR_COL_NUM >> 3),
        UNET_PAR_20_CONV2D_TRANSPOSE_IN_FILTER, UNET_PAR_20_CONV2D_TRANSPOSE_OUT_FILTER, UNET_PAR_20_CONV2D_TRANSPOSE_KERNEL_Y, 2, 1,
        UNET_PAR_22_CONV2D_IN_FILTER, UNET_PAR_22_CONV2D_OUT_FILTER, UNET_PAR_22_CONV2D_KERNEL_Y, 1, 1,
        (float *)layer_20_conv2d_transpose_weight, (float *)layer_20_conv2d_transpose_bias,
        (float *)layer_21_batch_norm_gamma, (float *)layer_21_batch_norm_beta,
        (float *)layer_21_batch_norm_mean, (float *)layer_21_batch_norm_variance,
        (float *)layer_22_conv2d_weight, (float *)layer_22_conv2d_bias,
        (float *)layer_23_batch_norm_gamma, (float *)layer_23_batch_norm_beta,
        (float *)layer_23_batch_norm_mean, (float *)layer_23_batch_norm_variance
    );
    float ****d1 = alloc_4d_arr(batch_size, ((UNET_PAR_ROW_NUM >> 2) << 1), ((UNET_PAR_COL_NUM >> 2) << 1), UNET_PAR_26_CONV2D_OUT_FILTER);
    decode_block(d0, x1, d1, batch_size, (UNET_PAR_ROW_NUM >> 2), (UNET_PAR_COL_NUM >> 2),
        UNET_PAR_24_CONV2D_TRANSPOSE_IN_FILTER, UNET_PAR_24_CONV2D_TRANSPOSE_OUT_FILTER, UNET_PAR_24_CONV2D_TRANSPOSE_KERNEL_Y, 2, 1,
        UNET_PAR_26_CONV2D_IN_FILTER, UNET_PAR_26_CONV2D_OUT_FILTER, UNET_PAR_26_CONV2D_KERNEL_Y, 1, 1,
        (float *)layer_24_conv2d_transpose_weight, (float *)layer_24_conv2d_transpose_bias,
        (float *)layer_25_batch_norm_gamma, (float *)layer_25_batch_norm_beta,
        (float *)layer_25_batch_norm_mean, (float *)layer_25_batch_norm_variance,
        (float *)layer_26_conv2d_weight, (float *)layer_26_conv2d_bias,
        (float *)layer_27_batch_norm_gamma, (float *)layer_27_batch_norm_beta,
        (float *)layer_27_batch_norm_mean, (float *)layer_27_batch_norm_variance
    );
    float ****d2 = alloc_4d_arr(batch_size, ((UNET_PAR_ROW_NUM >> 1) << 1), ((UNET_PAR_COL_NUM >> 1) << 1), UNET_PAR_30_CONV2D_OUT_FILTER);
    decode_block(d1, x0, d2, batch_size, (UNET_PAR_ROW_NUM >> 1), (UNET_PAR_COL_NUM >> 1),
        UNET_PAR_28_CONV2D_TRANSPOSE_IN_FILTER, UNET_PAR_28_CONV2D_TRANSPOSE_OUT_FILTER, UNET_PAR_28_CONV2D_TRANSPOSE_KERNEL_Y, 2, 1,
        UNET_PAR_30_CONV2D_IN_FILTER, UNET_PAR_30_CONV2D_OUT_FILTER, UNET_PAR_30_CONV2D_KERNEL_Y, 1, 1,
        (float *)layer_28_conv2d_transpose_weight, (float *)layer_28_conv2d_transpose_bias,
        (float *)layer_29_batch_norm_gamma, (float *)layer_29_batch_norm_beta,
        (float *)layer_29_batch_norm_mean, (float *)layer_29_batch_norm_variance,
        (float *)layer_30_conv2d_weight, (float *)layer_30_conv2d_bias,
        (float *)layer_31_batch_norm_gamma, (float *)layer_31_batch_norm_beta,
        (float *)layer_31_batch_norm_mean, (float *)layer_31_batch_norm_variance
    );
    float ****d3 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM, UNET_PAR_32_CONV2D_OUT_FILTER);
    conv2d(d2, d3, (float *)layer_32_conv2d_weight, (float *)layer_32_conv2d_bias, batch_size, UNET_PAR_32_CONV2D_IN_FILTER, UNET_PAR_32_CONV2D_OUT_FILTER, UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM, UNET_PAR_32_CONV2D_KERNEL_Y, 1, 0);
    resize_bilinear_batch(d3, output, batch_size, UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM, CMAQ_PAR_ROW_NUM, CMAQ_PAR_COL_NUM, UNET_PAR_32_CONV2D_OUT_FILTER);
    free_4d_arr(d0, batch_size, (UNET_PAR_ROW_NUM >> 3) << 1, (UNET_PAR_COL_NUM >> 3) << 1);
    free_4d_arr(d1, batch_size, (UNET_PAR_ROW_NUM >> 2), (UNET_PAR_COL_NUM >> 2));
    free_4d_arr(d2, batch_size, (UNET_PAR_ROW_NUM >> 1), (UNET_PAR_COL_NUM >> 1));
    free_4d_arr(d3, batch_size, UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM);
    return;
}

void forward(float ***input, float ****output, int batch_size)
{
    float ****gridded_input = alloc_4d_arr(batch_size, CMAQ_PAR_ROW_NUM, CMAQ_PAR_COL_NUM, CMAQ_PAR_SECTOR_NUM);
    float ****resized_gridded_input = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM, CMAQ_PAR_SECTOR_NUM);
    gridding(input, gridded_input, batch_size, CMAQ_PAR_REGION_NUM, CMAQ_PAR_SECTOR_NUM);
    resize_bilinear_batch(gridded_input, resized_gridded_input, batch_size, CMAQ_PAR_ROW_NUM, CMAQ_PAR_COL_NUM, UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM, CMAQ_PAR_SECTOR_NUM);
    float ****x0 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM, UNET_PAR_2_CONV2D_OUT_FILTER);
    float ****x1 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM >> 1, UNET_PAR_COL_NUM >> 1, UNET_PAR_6_CONV2D_OUT_FILTER);
    float ****x2 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM >> 2, UNET_PAR_COL_NUM >> 2, UNET_PAR_10_CONV2D_OUT_FILTER);
    float ****x3 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM >> 3, UNET_PAR_COL_NUM >> 3, UNET_PAR_12_CONV2D_OUT_FILTER);
    float ****x4 = alloc_4d_arr(batch_size, UNET_PAR_ROW_NUM >> 3, UNET_PAR_COL_NUM >> 3, UNET_PAR_16_CONV2D_OUT_FILTER);
    encoding(resized_gridded_input, x0, x1, x2, x3, x4, batch_size);
    decoding(x4, output, x0, x1, x2, x3, batch_size);
    // In: (CMAQ_PAR_ROW_NUM, CMAQ_PAR_COL_NUM, 1) -> Out: 역변환된 prediction
    /*
    for (int b = 0; b < batch_size; b++) {
        for (int r = 0; r < CMAQ_PAR_ROW_NUM; r++) {
            for (int c = 0; c < CMAQ_PAR_COL_NUM; c++) {
                int idx = r * CMAQ_PAR_COL_NUM + c;
                float val = fabsf(output[b][r][c][0]);
                output[b][r][c][0] = val * PREDICTION_SCALE[idx] + PREDICTION_MEAN[idx];
            }
        }
    }
    */
    free_4d_arr(gridded_input, batch_size, CMAQ_PAR_ROW_NUM, CMAQ_PAR_COL_NUM);
    free_4d_arr(resized_gridded_input, batch_size, UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM);
    free_4d_arr(x0, batch_size, UNET_PAR_ROW_NUM, UNET_PAR_COL_NUM);
    free_4d_arr(x1, batch_size, UNET_PAR_ROW_NUM >> 1, UNET_PAR_COL_NUM >> 1);
    free_4d_arr(x2, batch_size, UNET_PAR_ROW_NUM >> 2, UNET_PAR_COL_NUM >> 2);
    free_4d_arr(x3, batch_size, UNET_PAR_ROW_NUM >> 3, UNET_PAR_COL_NUM >> 3);
    free_4d_arr(x4, batch_size, UNET_PAR_ROW_NUM >> 3, UNET_PAR_COL_NUM >> 3);
}

void call(float *c_inputs, float *c_outputs, int batch_size)
{
    float ***inputs = alloc_3d_arr(batch_size, CMAQ_PAR_REGION_NUM, CMAQ_PAR_SECTOR_NUM);
    float ****outputs = alloc_4d_arr(batch_size, CMAQ_PAR_ROW_NUM, CMAQ_PAR_COL_NUM, UNET_PAR_32_CONV2D_OUT_FILTER);
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < CMAQ_PAR_REGION_NUM; j++) {
            for (size_t k = 0; k < CMAQ_PAR_SECTOR_NUM; k++) {
                inputs[i][j][k] = c_inputs[i * CMAQ_PAR_REGION_NUM * CMAQ_PAR_SECTOR_NUM + j * CMAQ_PAR_SECTOR_NUM + k];
            }
        }
    }
    forward(inputs, outputs, batch_size);
    /*
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < CMAQ_PAR_ROW_NUM; j++) {
            for (size_t k = 0; k < CMAQ_PAR_COL_NUM; k++) {
                printf("%.4f ", outputs[i][j][k][0]);
            }
            printf("\n");
        }
        printf("\n");
    }
    */
    free_3d_arr(inputs, batch_size, CMAQ_PAR_REGION_NUM);
    free_4d_arr(outputs, batch_size, CMAQ_PAR_ROW_NUM, CMAQ_PAR_COL_NUM);
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
    /* 
    for (int b = 0; b < batch_size; b++) {
        for (int r = 0; r < CMAQ_PAR_ROW_NUM; r++) {
            for (int c = 0; c < CMAQ_PAR_COL_NUM; c++) {
                printf("%.4f ", outputs[b][r][c][0]);
            }
            printf("\n");
        }
        printf("\n");
    }
    */
    free_3d_arr(inputs, batch_size, CMAQ_PAR_REGION_NUM);
    free_4d_arr(outputs, batch_size, CMAQ_PAR_ROW_NUM, CMAQ_PAR_COL_NUM);
}
