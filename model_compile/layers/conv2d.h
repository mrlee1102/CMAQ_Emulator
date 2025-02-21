#ifndef OPT_CONV2D_H
#define OPT_CONV2D_H

#include "utils.h"

void conv2d(
    float ****input, float ****output,
    const float *weights, const float *bias,
    int batch_size, int in_channels, int out_channels,
    int in_height, int in_width, int kernel_size, int stride, int padding) {
    int out_height = (in_height - kernel_size + 2 * padding) / stride + 1;
    int out_width = (in_width - kernel_size + 2 * padding) / stride + 1;

    // Iterate over each sample in the batch
    for (int b = 0; b < batch_size; b++) {
        // Iterate over each output channel
        for (int oc = 0; oc < out_channels; oc++) {
            // Iterate over the output spatial dimensions
            for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                    // Initialize the output value with the bias
                    output[b][oh][ow][oc] = bias[oc];

                    // Iterate over the input channels
                    for (int ic = 0; ic < in_channels; ic++) {
                        // Iterate over the kernel spatial dimensions
                        for (int kh = 0; kh < kernel_size; kh++) {
                            for (int kw = 0; kw < kernel_size; kw++) {
                                // Calculate the input spatial indices
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;

                                // Check if the input indices are within bounds
                                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                    // Calculate the index in the flattened weights array
                                    int weight_index = ((oc * in_channels + ic) * kernel_size + kw) * kernel_size + kh;

                                    // Accumulate the convolution result
                                    output[b][oh][ow][oc] += input[b][ih][iw][ic] * weights[weight_index];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return;
}

#endif