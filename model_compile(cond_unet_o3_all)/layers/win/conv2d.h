#ifndef OPT_CONV2D_H
#define OPT_CONV2D_H

#include "utils.h"

DWORD WINAPI process_conv2d(LPVOID params) {
    Conv2DThreadData *p = (Conv2DThreadData *)params;

    for (int b = 0; b < p->batch_size; b++) {
        for (int oc = p->startChannel; oc < p->endChannel; oc++) {
            for (int oh = 0; oh < p->out_height; oh++) {
                for (int ow = 0; ow < p->out_width; ow++) {
                    p->output[b][oh][ow][oc] = p->bias[oc];
                    for (int ic = 0; ic < p->in_channels; ic++) {
                        for (int kh = 0; kh < p->kernel_size; kh++) {
                            for (int kw = 0; kw < p->kernel_size; kw++) {
                                int ih = oh * p->stride - p->padding + kh;
                                int iw = ow * p->stride - p->padding + kw;
                                if (ih >= 0 && ih < p->in_height && iw >= 0 && iw < p->in_width) {
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

void conv2d(
    float ****input, float ****output,
    const float *weights, const float *bias,
    int batch_size, int in_channels, int out_channels,
    int in_height, int in_width, int kernel_size, int stride, int padding) {

    int numThreads = getNumberOfCores(); // Or based on the number of available CPU cores
    HANDLE *handles = malloc(numThreads * sizeof(HANDLE));
    Conv2DThreadData *threadData = malloc(numThreads * sizeof(Conv2DThreadData));

    int out_height = (in_height - kernel_size + 2 * padding) / stride + 1;
    int out_width = (in_width - kernel_size + 2 * padding) / stride + 1;
    int channelsPerThread = out_channels / numThreads;
    int remainingChannels = out_channels % numThreads;

    for (int i = 0; i < numThreads; i++) {
        int startChannel = i * channelsPerThread;
        int endChannel = startChannel + channelsPerThread + (i == numThreads - 1 ? remainingChannels : 0);
        threadData[i] = (Conv2DThreadData){input, output, weights, bias, startChannel, endChannel,
                                         batch_size, in_channels, out_channels, in_height, in_width,
                                         out_height, out_width, kernel_size, stride, padding};
        handles[i] = CreateThread(NULL, 0, process_conv2d, &threadData[i], 0, NULL);
    }

    WaitForMultipleObjects(numThreads, handles, TRUE, INFINITE);
    for (int i = 0; i < numThreads; i++) {
        CloseHandle(handles[i]);
    }
    free(handles);
    free(threadData);
}


#endif