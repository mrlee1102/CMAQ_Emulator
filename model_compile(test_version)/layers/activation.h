#ifndef OPT_ACT_H
#define OPT_ACT_H

#include "utils.h"

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

void silu(
    float ****input, int batch_size, int height, int width, int channels) {
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                for (int c = 0; c < channels; c++) {
                    float x = input[b][h][w][c];
                    input[b][h][w][c] = x * sigmoid(x);
                }
            }
        }
    }
}

void relu(
    float ****input, int batch_size, int height, int width, int channels){
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                for (int c = 0; c < channels; c++) {
                    input[b][h][w][c] = c_max(0.0, input[b][h][w][c]);
                }
            }
        }
    }
}

#endif // OPT_ACT_H