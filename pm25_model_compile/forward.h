#ifndef FORWARD_H
#define FORWARD_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

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
#elif defined(__unix__)
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

/* Parameter headers */
#include "params/params.h"
#include "params/layer_gridding.h"

#include "params/layer_0_params_dense.h"
#include "params/layer_1_params_conv2d.h"
#include "params/layer_2_params_dense.h"
#include "params/layer_3_params_batch_norm.h"
#include "params/layer_4_params_conv2d.h"
#include "params/layer_5_params_batch_norm.h"
#include "params/layer_6_params_conv2d.h"
#include "params/layer_7_params_dense.h"
#include "params/layer_8_params_batch_norm.h"
#include "params/layer_9_params_conv2d.h"
#include "params/layer_10_params_batch_norm.h"
#include "params/layer_11_params_conv2d.h"
#include "params/layer_12_params_dense.h"
#include "params/layer_13_params_batch_norm.h"
#include "params/layer_14_params_conv2d.h"
#include "params/layer_15_params_batch_norm.h"
#include "params/layer_16_params_conv2d.h"
#include "params/layer_17_params_dense.h"
#include "params/layer_18_params_batch_norm.h"
#include "params/layer_19_params_conv2d.h"
#include "params/layer_20_params_batch_norm.h"
#include "params/layer_21_params_conv2d.h"
#include "params/layer_22_params_dense.h"
#include "params/layer_23_params_batch_norm.h"
#include "params/layer_24_params_conv2d.h"
#include "params/layer_25_params_batch_norm.h"
#include "params/layer_26_params_conv2d_transpose.h"
#include "params/layer_27_params_batch_norm.h"
#include "params/layer_28_params_dense.h"
#include "params/layer_29_params_conv2d.h"
#include "params/layer_30_params_batch_norm.h"
#include "params/layer_31_params_conv2d_transpose.h"
#include "params/layer_32_params_batch_norm.h"
#include "params/layer_33_params_dense.h"
#include "params/layer_34_params_conv2d.h"
#include "params/layer_35_params_batch_norm.h"
#include "params/layer_36_params_conv2d_transpose.h"
#include "params/layer_37_params_batch_norm.h"
#include "params/layer_38_params_dense.h"
#include "params/layer_39_params_conv2d.h"
#include "params/layer_40_params_batch_norm.h"
#include "params/layer_41_params_conv2d.h"

void encoding(
    float ****input,
    float ****x0, float ****x1, float ****x2, float ****x3, float ****x4,
    float **bc_input,
    int batch_size
);

void decoding(
    float ****bottleneck,
    float ****output,
    float ****x0,
    float ****x1,
    float ****x2,
    float ****x3,
    float **bc_input,
    int batch_size
);

void forward(float ***input, float **bc_input, float ****output, int batch_size);
void call(float *c_inputs, float *c_bc_inputs, float *c_outputs, int batch_size);

float *set_sector(void);
float ***set_ctrl_mat(int batch_size);

/* 새로운 타입 정의 */
typedef enum {
    EMBEDDING_NORMAL,
    EMBEDDING_TIME
} EmbeddingType;

typedef enum {
    ACTIVATION_SILU,
    ACTIVATION_RELU,
    ACTIVATION_TANH,
    ACTIVATION_SIGMOID
} ActivationType;

/* 임베딩 관련 함수 */
void time_embedding_1d(const float *input, int input_dim, int emb_dim, float *output);
void normal_embedding_1d(const float *input, int input_dim, int emb_dim, float *output);
void embedding_forward(const float *input, int input_dim, int emb_dim, EmbeddingType type, float *output);

/* 활성화 함수 */
void apply_activation(float *tensor, int size, ActivationType type);

/* 조건부 주입 관련 함수 */
void conditional_injection(
    float ****feature,
    float *emb,
    int batch_size,
    int height,
    int width,
    int channels,
    ActivationType activation_type
);

/* 수정된 인코더/디코더 블록 */
void encode_block_conditional(
    float ****input, float ****output_skip, const float *emb,
    int batch_size, int height, int width,
    int f0, int f1, int k0, int s0, int p0,
    int f2, int f3, int k1, int s1, int p1,
    const float *w0, const float *b0,
    const float *g0, const float *b_0, const float *m0, const float *v0,
    const float *w1, const float *b1,
    const float *g1, const float *b_1, const float *m1, const float *v1
);

void decode_block_conditional(
    float ****in0, float ****in1, float ****output, const float *emb,
    int batch_size, int height, int width,
    int f0, int f1, int k0, int s0, int p0,
    int f2, int f3, int k1, int s1, int p1,
    const float *wT, const float *bT,
    const float *g0, const float *b_0, const float *m0, const float *v0,
    const float *w1, const float *b1,
    const float *g1, const float *b_1, const float *m1, const float *v1
);

#endif /* FORWARD_H */
