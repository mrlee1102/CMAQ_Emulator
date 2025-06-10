#ifndef OPT_UTILS_H
#define OPT_UTILS_H

#include <math.h>
#include <windows.h>
#include <stdio.h>

#define NUM_THREAD 8
#define c_max(a, b) ((a) > (b) ? (a) : (b))
#define c_min(a, b) ((a) < (b) ? (a) : (b))
#define c_abs(a) ((a) >= 0.0 ? (a) : -(a)) 

int getNumberOfCores() {
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return sysinfo.dwNumberOfProcessors;
}

typedef struct {
    float ****input;
    int startBatch;
    int endBatch;
    int height;
    int width;
    int channels;
} ActThreadData;

typedef struct {
    float ****input_0;
    float ****input_1;
    float ****output;
    int startBatch;
    int endBatch;
    int height;
    int width;
    int channels_0;
    int channels_1;
} ConcatThreadData;

typedef struct {
    float ****input;
    const float *gamma;
    const float *beta;
    const float *mean;
    const float *variance;
    int startBatch;
    int endBatch;
    int height;
    int width;
    int channels;
    float epsilon;
    float momentum;
} BNormThreadData;

typedef struct {
    float ****input;
    float ****output;
    const float *weights;
    const float *bias;
    int startChannel;
    int endChannel;
    int batch_size;
    int in_channels;
    int out_channels;
    int in_height;
    int in_width;
    int out_height;
    int out_width;
    int kernel_size;
    int stride;
    int padding;
} Conv2DThreadData;

typedef struct {
    float ****input;
    float ****output;
    const float *weights;
    const float *bias;
    int startBatch;
    int endBatch;
    int in_channels;
    int out_channels;
    int in_height;
    int in_width;
    int out_height;
    int out_width;
    int kernel_size;
    int stride;
    int padding;
} Conv2DTransposeThreadData;

#endif