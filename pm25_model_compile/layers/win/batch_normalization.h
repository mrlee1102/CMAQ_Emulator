#ifndef OPT_BNORM_H
#define OPT_BNORM_H

#include "utils.h"

DWORD WINAPI process_batch_normalization(LPVOID lpParam) {
    BNormThreadData *data = (BNormThreadData*)lpParam;
    for (int b = data->startBatch; b < data->endBatch; b++) {
        for (int h = 0; h < data->height; h++) {
            for (int w = 0; w < data->width; w++) {
                for (int c = 0; c < data->channels; c++) {
                    float normalized = (data->input[b][h][w][c] - data->mean[c]) / sqrt(data->variance[c] + data->epsilon);
                    data->input[b][h][w][c] = data->gamma[c] * normalized + data->beta[c];
                }
            }
        }
    }
    return 0;
}

void batch_normalization(
    float ****input, const float *gamma, const float *beta,
    const float *mean, const float *variance, int batch_size,
    int height, int width, int channels, float epsilon, float momentum) {
    int numThreads = getNumberOfCores();  // Choose an appropriate number of threads based on your hardware
    HANDLE *handles = malloc(numThreads * sizeof(HANDLE));
    BNormThreadData *threadData = malloc(numThreads * sizeof(BNormThreadData));

    int batchSize = batch_size / numThreads;
    int remainingBatches = batch_size % numThreads;

    for (int i = 0; i < numThreads; i++) {
        int startBatch = i * batchSize;
        int endBatch = (i + 1) * batchSize + (i == numThreads - 1 ? remainingBatches : 0);
        threadData[i] = (BNormThreadData){
            input, gamma, beta, mean, variance,
            startBatch, endBatch, height, width, channels, epsilon, momentum};
        handles[i] = CreateThread(NULL, 0, process_batch_normalization, &threadData[i], 0, NULL);
    }

    // Wait for all threads to complete
    WaitForMultipleObjects(numThreads, handles, TRUE, INFINITE);

    // Cleanup
    for (int i = 0; i < numThreads; i++) {
        CloseHandle(handles[i]);
    }
    free(handles);
    free(threadData);
    return;
}

#endif // OPT_BNORM_H