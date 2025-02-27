#ifndef OPT_CONCAT_H
#define OPT_CONCAT_H

#include "utils.h"

void concatenate(
    float ****input_0, float ****input_1, float ****output,
    int batch_size, int height, int width, int channels_0, int channels_1) {
    int out_channels = channels_0 + channels_1;
    // Iterate over each sample in the batch
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < batch_size; b++) {
        // Iterate over the spatial dimensions
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                // Copy the channels from input_0 to the output
                for (int c = 0; c < channels_0; c++) {
                    output[b][h][w][c] = input_0[b][h][w][c];
                }

                // Copy the channels from input_1 to the output
                for (int c = 0; c < channels_1; c++) {
                    output[b][h][w][channels_0 + c] = input_1[b][h][w][c];
                }
            }
        }
    }
}

#endif