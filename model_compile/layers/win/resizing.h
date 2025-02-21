#ifndef OPT_RESIZE_H
#define OPT_RESIZE_H

#include "utils.h"

void resize_bilinear(
    float ***input_image, float ***output_image,
    int input_height, int input_width,
    int output_height, int output_width,
    int channels) {
    for (int c = 0; c < channels; c++) {
        for (int y = 0; y < output_height; y++) {
            for (int x = 0; x < output_width; x++) {
                float input_y = (y + 0.5) * (float)input_height / output_height - 0.5;
                float input_x = (x + 0.5) * (float)input_width / output_width - 0.5;
                
                int y0 = (int)floor(input_y);
                int x0 = (int)floor(input_x);
                int y1 = y0 + 1;
                int x1 = x0 + 1;
                
                float dy = input_y - y0;
                float dx = input_x - x0;
                
                float w00 = (1 - dy) * (1 - dx);
                float w01 = (1 - dy) * dx;
                float w10 = dy * (1 - dx);
                float w11 = dy * dx;
                
                y0 = c_max(0, c_min(y0, input_height - 1));
                x0 = c_max(0, c_min(x0, input_width - 1));
                y1 = c_max(0, c_min(y1, input_height - 1));
                x1 = c_max(0, c_min(x1, input_width - 1));
                
                float value = 0;
                value += input_image[y0][x0][c] * w00;
                value += input_image[y0][x1][c] * w01;
                value += input_image[y1][x0][c] * w10;
                value += input_image[y1][x1][c] * w11;
                
                output_image[y][x][c] = value;
            }
        }
    }
}

void resize_bilinear_batch(
    float ****input_image, float ****output_image,
    int batch_size, int input_height, int input_width,
    int output_height, int output_width,
    int channels) {
    for (int b = 0; b < batch_size; b++) {
        resize_bilinear(
            input_image[b], output_image[b],
            input_height, input_width, output_height, output_width, channels);
    }
}

#endif