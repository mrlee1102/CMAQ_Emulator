#ifndef OPT_BNORM_H
#define OPT_BNORM_H

#include "utils.h"

void batch_normalization(
    float ****input, const float *gamma, const float *beta,
    const float *mean, const float *variance, int batch_size,
    int height, int width, int channels, float epsilon, float momentum) {
    #pragma omp parallel for collapse(4)
    for (int b = 0; b < batch_size; b++) {         // Loop over each batch
        for (int h = 0; h < height; h++) {         // Loop over each height
            for (int w = 0; w < width; w++) {      // Loop over each width
                for (int c = 0; c < channels; c++) { // Loop over each channel
                    // Normalize each element
                    float normalized = (input[b][h][w][c] - mean[c]) / sqrt(variance[c] + epsilon);
                    // Scale and shift
                    input[b][h][w][c] = gamma[c] * normalized + beta[c];
                }
            }
        }
    }
}

#endif // OPT_BNORM_H