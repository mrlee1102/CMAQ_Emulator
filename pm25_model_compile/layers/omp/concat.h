#ifndef OPT_CONCAT_H
#define OPT_CONCAT_H

#include "utils.h"
#include <stdio.h>
#include <omp.h>

/**
 * 4D 텐서 두 개를 채널 축으로 이어 붙입니다.
 *
 * input_0: [batch_size][H][W][channels_0]
 * input_1: [batch_size][H][W][channels_1]
 * output:  [batch_size][H][W][channels_0 + channels_1]
 *
 * 여기서 H,W는 “출력 공간 해상도”를 가리키므로,
 * decode_block_conditional()에서는 업샘플된 크기(oh,ow)를 넘깁니다.
 */
void concatenate(
    float ****input_0,
    float ****input_1,
    float ****output,
    int batch_size,
    int height,      // = 출력 높이(oh)
    int width,       // = 출력 너비(ow)
    int channels_0,  // 첫 입력의 채널 수
    int channels_1   // 두 번째 입력의 채널 수
) {
    printf("[DEBUG concat] H=%d, W=%d, ch0=%d, ch1=%d\n", height, width, channels_0, channels_1);
    
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                // input_0 쪽 복사
                for (int c = 0; c < channels_0; c++) {
                    output[b][h][w][c] = input_0[b][h][w][c];
                }
                // input_1 쪽 복사
                for (int c = 0; c < channels_1; c++) {
                    output[b][h][w][channels_0 + c]
                        = input_1[b][h][w][c];
                }
            }
        }
    }
}

#endif  // OPT_CONCAT_H
