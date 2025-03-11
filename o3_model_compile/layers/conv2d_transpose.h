#ifndef OPT_CONV2D_T_H
#define OPT_CONV2D_T_H

#include "utils.h"

void conv2d_transpose(
    float ****input, float ****output,
    const float *weights, const float *bias,
    int batch_size, int in_channels, int out_channels,
    int in_height, int in_width, int kernel_size, int stride, int padding) {
    // Adjusting the dimensions to match height-first in calculations
    int out_height = in_height * stride;
    int out_width = in_width * stride;

    // Iterate over each sample in the batch
    for (int b = 0; b < batch_size; b++) {
        // Iterate over each output channel
        for (int oc = 0; oc < out_channels; oc++) {
            // Initialize the output tensor with the bias
            for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                    output[b][oh][ow][oc] = bias[oc];
                }
            }

            // Iterate over the input channels
            for (int ic = 0; ic < in_channels; ic++) {
                // Iterate over the input spatial dimensions
                for (int ih = 0; ih < in_height; ih++) {
                    for (int iw = 0; iw < in_width; iw++) {
                        // Iterate over the kernel spatial dimensions
                        for (int kh = 0; kh < kernel_size; kh++) {
                            for (int kw = 0; kw < kernel_size; kw++) {
                                // Calculate the output spatial indices
                                int oh = ih * stride + kh - padding;
                                int ow = iw * stride + kw - padding;

                                // Check if the output indices are within bounds
                                if (oh >= 0 && oh < out_height && ow >= 0 && ow < out_width) {
                                    // Calculate the index in the flattened weights array
                                    int weight_index = ((oc * in_channels + ic) * kernel_size + kw) * kernel_size + kh;

                                    // Accumulate the transposed convolution result
                                    output[b][oh][ow][oc] += input[b][ih][iw][ic] * weights[weight_index];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#endif