#ifndef OPT_CONV2D_T_H
#define OPT_CONV2D_T_H

#include "utils.h"

DWORD WINAPI process_conv2d_transpose(LPVOID params) {
    Conv2DTransposeThreadData *p = (Conv2DTransposeThreadData *)params;
    for (int b = p->startBatch; b < p->endBatch; b++) {
        for (int oc = 0; oc < p->out_channels; oc++) {
            for (int oh = 0; oh < p->out_height; oh++) {
                for (int ow = 0; ow < p->out_width; ow++) {
                    p->output[b][oh][ow][oc] = p->bias[oc];
                }
            }

            for (int ic = 0; ic < p->in_channels; ic++) {
                for (int ih = 0; ih < p->in_height; ih++) {
                    for (int iw = 0; iw < p->in_width; iw++) {
                        for (int kh = 0; kh < p->kernel_size; kh++) {
                            for (int kw = 0; kw < p->kernel_size; kw++) {
                                int oh = ih * p->stride + kh - p->padding;
                                int ow = iw * p->stride + kw - p->padding;
                                if (oh >= 0 && oh < p->out_height && ow >= 0 && ow < p->out_width) {
                                    int weight_index = ((oc * p->in_channels + ic) * p->kernel_size + kh) * p->kernel_size + kw;
                                    p->output[b][oh][ow][oc] += p->input[b][ih][iw][ic] * p->weights[weight_index];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return 0;
}

void conv2d_transpose(float ****input, float ****output, const float *weights, const float *bias, 
                             int batch_size, int in_channels, int out_channels,
                             int in_height, int in_width, int kernel_size, int stride, int padding) {
    int out_height = in_height * stride;
    int out_width = in_width * stride;

    int num_threads = getNumberOfCores();  // Customize based on the system's capability
    HANDLE *threads = malloc(num_threads * sizeof(HANDLE));
    Conv2DTransposeThreadData *threadData = malloc(num_threads * sizeof(Conv2DTransposeThreadData));

    int batches_per_thread = batch_size / num_threads;
    int remaining_batches = batch_size % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int startBatch = i * batches_per_thread;
        int endBatch = (i + 1) * batches_per_thread + (i == num_threads - 1 ? remaining_batches : 0);
        threadData[i] = (Conv2DTransposeThreadData){input, output, weights, bias, startBatch, endBatch, in_channels, out_channels,
                                                 in_height, in_width, out_height, out_width, kernel_size, stride, padding};
        threads[i] = CreateThread(NULL, 0, process_conv2d_transpose, &threadData[i], 0, NULL);
    }

    // Wait for all threads to complete
    WaitForMultipleObjects(num_threads, threads, TRUE, INFINITE);

    for (int i = 0; i < num_threads; i++) {
        CloseHandle(threads[i]);
    }
    free(threads);
    free(threadData);
}


#endif