#ifndef UTILS_MEMORY_H
#define UTILS_MEMORY_H

#include <stdlib.h>

float ****alloc_4d_arr(int dim0, int dim1, int dim2, int dim3) {
    float ****arr = (float ****)malloc(dim0 * sizeof(float ***));
    for (int i = 0; i < dim0; i++) {
        arr[i] = (float ***)malloc(dim1 * sizeof(float **));
        for (int j = 0; j < dim1; j++) {
            arr[i][j] = (float **)malloc(dim2 * sizeof(float *));
            for (int k = 0; k < dim2; k++) {
                arr[i][j][k] = (float *)malloc(dim3 * sizeof(float));
            }
        }
    }

    for (int i = 0; i < dim0; i++) {
        for (int j = 0; j < dim1; j++) {
            for (int k = 0; k < dim2; k++) {
                for (int l = 0; l < dim3; l++) {
                    arr[i][j][k][l] = 0.0;
                }
            }
        }
    }
    return arr;
}

float ***alloc_3d_arr(int dim0, int dim1, int dim2) {
    float ***arr = (float ***)malloc(dim0 * sizeof(float **));
    for (int i = 0; i < dim0; i++) {
        arr[i] = (float **)malloc(dim1 * sizeof(float *));
        for (int j = 0; j < dim1; j++) {
            arr[i][j] = (float *)malloc(dim2 * sizeof(float));
        }
    }

    for (int i = 0; i < dim0; i++) {
        for (int j = 0; j < dim1; j++) {
            for (int k = 0; k < dim2; k++) {
                arr[i][j][k] = 0.0;
            }
        }
    }
    return arr;
}

float **alloc_2d_arr(int dim0, int dim1) {
    float **arr = (float **)malloc(dim0 * sizeof(float *));
    for (int i = 0; i < dim0; i++) {
        arr[i] = (float *)malloc(dim1 * sizeof(float));
    }

    for (int i = 0; i < dim0; i++) {
        for (int j = 0; j < dim1; j++) {
            arr[i][j] = 0.0;
        }
    }
    return arr;
}

void free_4d_arr(float ****arr, int dim0, int dim1, int dim2) {
    for (int i = 0; i < dim0; i++) {
        for (int j = 0; j < dim1; j++) {
            for (int k = 0; k < dim2; k++) {
                free(arr[i][j][k]);
            }
            free(arr[i][j]);
        }
        free(arr[i]);
    }
    free(arr);
}

void free_3d_arr(float ***arr, int dim0, int dim1) {
    for (int i = 0; i < dim0; i++) {
        for (int j = 0; j < dim1; j++) {
            free(arr[i][j]);
        }
        free(arr[i]);
    }
    free(arr);
}

void free_2d_arr(float **arr, int dim0) {
    for (int i = 0; i < dim0; i++) {
        free(arr[i]);
    }
    free(arr);
}

#endif