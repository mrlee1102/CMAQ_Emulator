#ifndef MAIN_FORWARD_H
#define MAIN_FORWARD_H

#include <stdio.h>
#include <stdlib.h>

// 필요에 따라 로거/메모리 함수 포함
#include "utils/logger.h"
#include "utils/mem.h"

// OS별 레이어 함수(OMP/OpenMP, Windows, etc.) 헤더
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

// 공통 params.h (UNET_PAR_XXX 등 매크로 정의)
#include "params/params.h"
#include "params/layer_gridding.h"
// 인코더(encoding) 측 레이어
#include "params/layer_0_params_conv2d.h"
#include "params/layer_1_params_conv2d.h"
#include "params/layer_2_params_batch_normalization.h"
#include "params/layer_3_params_conv2d.h"
#include "params/layer_4_params_batch_normalization.h"
#include "params/layer_5_params_conv2d.h"
#include "params/layer_6_params_conv2d.h"
#include "params/layer_7_params_batch_normalization.h"
#include "params/layer_8_params_conv2d.h"
#include "params/layer_9_params_batch_normalization.h"
#include "params/layer_10_params_conv2d.h"
#include "params/layer_11_params_conv2d.h"
#include "params/layer_12_params_batch_normalization.h"
#include "params/layer_13_params_conv2d.h"
#include "params/layer_14_params_batch_normalization.h"
#include "params/layer_15_params_conv2d.h"
#include "params/layer_16_params_conv2d.h"
#include "params/layer_17_params_batch_normalization.h"
#include "params/layer_18_params_conv2d.h"
#include "params/layer_19_params_batch_normalization.h"
#include "params/layer_20_params_conv2d.h"
#include "params/layer_21_params_conv2d.h"
#include "params/layer_22_params_batch_normalization.h"
#include "params/layer_23_params_conv2d.h"
#include "params/layer_24_params_batch_normalization.h"
#include "params/layer_25_params_conv2d_transpose.h"
#include "params/layer_26_params_batch_normalization.h"
#include "params/layer_27_params_conv2d.h"
#include "params/layer_28_params_conv2d.h"
#include "params/layer_29_params_batch_normalization.h"
#include "params/layer_30_params_conv2d_transpose.h"
#include "params/layer_31_params_batch_normalization.h"
#include "params/layer_32_params_conv2d.h"
#include "params/layer_33_params_conv2d.h"
#include "params/layer_34_params_batch_normalization.h"
#include "params/layer_35_params_conv2d_transpose.h"
#include "params/layer_36_params_batch_normalization.h"
#include "params/layer_37_params_conv2d.h"
#include "params/layer_38_params_conv2d.h"
#include "params/layer_39_params_batch_normalization.h"
#include "params/layer_40_params_conv2d_transpose.h"
#include "params/layer_41_params_batch_normalization.h"
#include "params/layer_42_params_conv2d.h"
#include "params/layer_43_params_conv2d.h"
#include "params/layer_44_params_batch_normalization.h"
#include "params/layer_45_params_conv2d.h"


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
);

void decode_block(
  float ****input_0, float ****input_1, float ****output,
  int batch_size, int height, int width,
  int filter_0, int filter_1, int kernel_0, int stride_0, int padding_0,
  int filter_2, int filter_3, int kernel_1, int stride_1, int padding_1,
  int filter_4, int filter_5, int kernel_2, int stride_2, int padding_2,
  const float *weight_0, const float *bias_0,
  const float *gamma_0, const float *beta_0, const float *mean_0, const float *var_0,
  const float *weight_1, const float *bias_1,
  const float *weight_2, const float *bias_2,
  const float *gamma_1, const float *beta_1, const float *mean_1, const float *var_1
);

void encoding(float ****input, float ****x0, float ****x1, float ****x2, float ****x3, float ****x4, int batch_size);
void decoding(float ****input, float ****output, float ****x0, float ****x1, float ****x2, float ****x3, int batch_size);
// void forward(float ***input, float ****output, int batch_size);
void forward(float ***ctrl_input, float ****meteo_input, float ****output, int batch_size);
// void call(float *c_inputs, float *c_outputs, int batch_size);
void call(float *c_ctrl_inputs, float *c_meteo_inputs, float *c_outputs, int batch_size);
float *set_sector(void);
float ***set_ctrl_mat(int batch_size);

#endif // MAIN_FORWARD_H