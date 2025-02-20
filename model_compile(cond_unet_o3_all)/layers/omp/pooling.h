#ifndef OPT_POOL_H
#define OPT_POOL_H

#include "utils.h"

void max_pool(
    float ****input, float ****output, int batch_size,
    int height, int width, int channels, int pool_size, int stride) {
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; b++) { // Iterate over each batch
        for (int c = 0; c < channels; c++) { // Iterate over each channel
            for (int h = 0; h < height; h += stride) { // Iterate over the input height
                for (int w = 0; w < width; w += stride) { // Iterate over the input width
                    float max_val = -3.40282e+38;
                    // Determine the boundaries of the pool
                    for (int ph = 0; ph < pool_size; ph++) { // Pool height
                        int current_h = h + ph;
                        if (current_h >= height) break; // Prevent reading out of bounds
                        for (int pw = 0; pw < pool_size; pw++) { // Pool width
                            int current_w = w + pw;
                            if (current_w >= width) break; // Prevent reading out of bounds
                            max_val = c_max(max_val, input[b][current_h][current_w][c]);
                        }
                    }
                    output[b][(int) h / stride][(int) w / stride][c] = max_val;
                }
            }
        }
    }
}


#endif // OPT_POOL_H