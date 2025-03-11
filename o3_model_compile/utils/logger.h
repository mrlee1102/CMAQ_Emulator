#ifndef UTILS_LOGGER_H
#define UTILS_LOGGER_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void logger(const char* tag, const char* message) {
    time_t now;
    time(&now);
    printf("%s [%s]: %s\n", ctime(&now), tag, message);
    return;
}

int save_to_csv(
    float ****output, const char *filename,
    int bidx, int height, int width, int cidx) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        return 1;
    }

    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            float value = output[bidx][h][w][cidx];
            if (value < 0) {
                fprintf(file, "%f, ", output[bidx][h][w][cidx]);
            } else {
                fprintf(file, " %f, ", output[bidx][h][w][cidx]);
            }
        }
        fprintf(file, "\n");
    }

    logger("CSV file generated", filename);
    return 0;
}

#endif