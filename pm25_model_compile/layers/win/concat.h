#ifndef OPT_CONCAT_H
#define OPT_CONCAT_H

#include "utils.h"

DWORD WINAPI process_concatenate(LPVOID params) {
    ConcatThreadData *p = (ConcatThreadData *)params;
    int out_channels = p->channels_0 + p->channels_1;

    for (int b = p->startBatch; b < p->endBatch; b++) {
        for (int h = 0; h < p->height; h++) {
            for (int w = 0; w < p->width; w++) {
                for (int c = 0; c < p->channels_0; c++) {
                    p->output[b][h][w][c] = p->input_0[b][h][w][c];
                }
                for (int c = 0; c < p->channels_1; c++) {
                    p->output[b][h][w][p->channels_0 + c] = p->input_1[b][h][w][c];
                }
            }
        }
    }
    return 0;
}

void concatenate(float ****input_0, float ****input_1, float ****output,
                         int batch_size, int height, int width, int channels_0, int channels_1) {
    int num_threads = getNumberOfCores();  // Adjust based on system capabilities and workload
    HANDLE *threads = malloc(num_threads * sizeof(HANDLE));
    ConcatThreadData *threadData = malloc(num_threads * sizeof(ConcatThreadData));

    int batches_per_thread = batch_size / num_threads;
    int remaining_batches = batch_size % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int startBatch = i * batches_per_thread;
        int endBatch = (i + 1) * batches_per_thread + (i == num_threads - 1 ? remaining_batches : 0);
        threadData[i] = (ConcatThreadData){input_0, input_1, output, startBatch, endBatch, height, width, channels_0, channels_1};
        threads[i] = CreateThread(NULL, 0, process_concatenate, &threadData[i], 0, NULL);
    }

    // Wait for all threads to complete
    WaitForMultipleObjects(num_threads, threads, TRUE, INFINITE);

    // Cleanup
    for (int i = 0; i < num_threads; i++) {
        CloseHandle(threads[i]);
    }
    free(threads);
    free(threadData);
}


#endif