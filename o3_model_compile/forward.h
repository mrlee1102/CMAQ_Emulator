#ifndef MAIN_FORWARD_H
#define MAIN_FORWARD_H

#include <stdio.h>
#include <stdlib.h>

#include "utils/logger.h"
#include "utils/mem.h"

#ifdef _WIN32
#include "layers/win/conv2d.h"
#include "layers/win/conv2d_transpose.h"
#include "layers/win/batch_normalization.h"
#include "layers/win/activation.h"
#include "layers/win/pooling.h"
#include "layers/win/resizing.h"
#include "layers/win/concat.h"
#elif unix
#include "layers/omp/conv2d.h"
#include "layers/omp/conv2d_transpose.h"
#include "layers/omp/batch_normalization.h"
#include "layers/omp/activation.h"
#include "layers/omp/pooling.h"
#include "layers/omp/resizing.h"
#include "layers/omp/concat.h"
#else
#include "layers/conv2d.h"
#include "layers/conv2d_transpose.h"
#include "layers/batch_normalization.h"
#include "layers/activation.h"
#include "layers/pooling.h"
#include "layers/resizing.h"
#include "layers/concat.h"
#endif

#include "params/params.h"
#include "params/layer_gridding.h"
#include "params/layer_0_params_conv2d.h"
#include "params/layer_1_params_batch_norm.h"
#include "params/layer_2_params_conv2d.h"
#include "params/layer_3_params_batch_norm.h"
#include "params/layer_4_params_conv2d.h"
#include "params/layer_5_params_batch_norm.h"
#include "params/layer_6_params_conv2d.h"
#include "params/layer_7_params_batch_norm.h"
#include "params/layer_8_params_conv2d.h"
#include "params/layer_9_params_batch_norm.h"
#include "params/layer_10_params_conv2d.h"
#include "params/layer_11_params_batch_norm.h"
#include "params/layer_12_params_conv2d.h"
#include "params/layer_13_params_batch_norm.h"
#include "params/layer_14_params_conv2d.h"
#include "params/layer_15_params_batch_norm.h"
#include "params/layer_16_params_conv2d.h"
#include "params/layer_17_params_batch_norm.h"
#include "params/layer_18_params_conv2d.h"
#include "params/layer_19_params_batch_norm.h"
#include "params/layer_20_params_conv2d_transpose.h"
#include "params/layer_21_params_batch_norm.h"
#include "params/layer_22_params_conv2d.h"
#include "params/layer_23_params_batch_norm.h"
#include "params/layer_24_params_conv2d_transpose.h"
#include "params/layer_25_params_batch_norm.h"
#include "params/layer_26_params_conv2d.h"
#include "params/layer_27_params_batch_norm.h"
#include "params/layer_28_params_conv2d_transpose.h"
#include "params/layer_29_params_batch_norm.h"
#include "params/layer_30_params_conv2d.h"
#include "params/layer_31_params_batch_norm.h"
#include "params/layer_32_params_conv2d.h"

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
);

void decode_block(
    float ****input_0, float ****input_1, float ****output, 
    int batch_size, int height, int width,
    int filter_0, int filter_1, int kernel_0, int stride_0, int padding_0,
    int filter_2, int filter_3, int kernel_1, int stride_1, int padding_1,
    const float *weight_0, const float *bias_0,
    const float *gamma_0, const float *beta_0, const float *mean_0, const float *var_0,
    const float *weight_1, const float *bias_1,
    const float *gamma_1, const float *beta_1, const float *mean_1, const float *var_1);

void encoding(float ****input, float ****x0, float ****x1, float ****x2, float ****x3, float ****x4, int batch_size);
// void decoding(float ****bottleneck, float ****output, float ****x0, float ****x1, float ****x2, int batch_size);
void decoding(
    float ****bottleneck, float ****output,
    float ****x0, float ****x1, float ****x2, float ****x3, int batch_size);
void forward(float ***input, float ****output, int batch_size);
void call(float *c_inputs, float *c_outputs, int batch_size);

float *set_sector(void);
float ***set_ctrl_mat(int batch_size);

#endif