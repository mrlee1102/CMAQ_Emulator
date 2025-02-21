#ifndef OPT_ACT_H
#define OPT_ACT_H

#include "utils.h"

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

DWORD WINAPI process_silu(LPVOID lpParam) {
    ActThreadData *data = (ActThreadData*)lpParam;
    for (int b = data->startBatch; b < data->endBatch; b++) {
        for (int h = 0; h < data->height; h++) {
            for (int w = 0; w < data->width; w++) {
                for (int c = 0; c < data->channels; c++) {
                    float x = data->input[b][h][w][c];
                    data->input[b][h][w][c] = x * sigmoid(x);
                }
            }
        }
    }
    return 0;
}

void silu(
    float ****input, int batch_size, int height, int width, int channels) {
    int numThreads = getNumberOfCores();  // Choose an appropriate number of threads based on your hardware
    HANDLE *handles = malloc(numThreads * sizeof(HANDLE));
    ActThreadData *threadData = malloc(numThreads * sizeof(ActThreadData));

    int batchSize = batch_size / numThreads;
    int remainingBatches = batch_size % numThreads;

    for (int i = 0; i < numThreads; i++) {
        int startBatch = i * batchSize;
        int endBatch = (i + 1) * batchSize + (i == numThreads - 1 ? remainingBatches : 0);
        threadData[i] = (ActThreadData){input, startBatch, endBatch, height, width, channels};
        handles[i] = CreateThread(NULL, 0, process_silu, &threadData[i], 0, NULL);
    }

    // Wait for all threads to complete
    WaitForMultipleObjects(numThreads, handles, TRUE, INFINITE);

    // Cleanup
    for (int i = 0; i < numThreads; i++) {
        CloseHandle(handles[i]);
    }
    free(handles);
    free(threadData);
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