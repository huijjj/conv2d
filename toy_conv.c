#include <time.h>
#include <stdint.h>

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <memory.h>
#include <math.h>

int _N;
int _IH;
int _IW; 
int _IC; 
int _OC;
int _KH; 
int _KW;

int32_t* _tensorIn;
int32_t* _kernel;

double benchmark(
    int32_t *tensorIn,
    int32_t *kernel,
    int32_t *tensorOut,
    int N,
    int IH, int IW, int IC, int OC,
    int KH, int KW
)
{
    int num_iter = 1; // originally 500

    struct timespec start, end;
    double total_time = 0;

    for (int eval_iter=0;eval_iter<num_iter;eval_iter++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        inference(tensorIn, kernel, tensorOut, N, IH, IW, IC, OC, KH, KW);
        clock_gettime(CLOCK_MONOTONIC, &end);

        total_time += (double)(end.tv_sec - start.tv_sec)*1000000 + (double)(end.tv_nsec - start.tv_nsec)/1000.0;
    }

    return total_time / (double)(num_iter);
}


// converts kernel into matrix, gets element in row r and column c of converted matrix
inline int32_t c_ker(int r, int c, int32_t* st) {
    return *(st + r * (_KH*_KW*_IC) + ((c % (_KH*_KW)) / _KW) * (_KW*_IC) + ((c % (_KH*_KW)) % _KW) * _IC + (c / (_KH*_KW)));
}

// converts input into matrix, gets element in row r and column c of converted matrix
inline int32_t c_in(int n, int r, int c, int32_t* st) {
    // element at (r , c) gets multiplied with (x, r) element in columnized kernel
    // and will be added up to element at (c / IH, c % IH, c), (H, W, C) at output

    // we want (h, w, ic) of original input
    // ic == r / (KH * KW)
    
    // first element should be ((c / IH) - (KH / 2), (c % IH) - (KW / 2))
    // r % IC is the index with in kernel
    const int h = (c / _IW) - (_KH / 2) + ((r % (_KH*_KW)) / _KW); 
    const int w = (c % _IW) - (_KW / 2) + ((r % (_KH*_KW)) % _KW);

    if(h < 0 || h >= _IH || w < 0 || w >= _IW) { // padding
        return 0;
    }
    else {
        return *(st + n * _IH*_IW*_IC + h * _IW*_IC + w * _IC + (r / (_KH*_KW)));
    }
}

int inference(
    int32_t *tensorIn,
    int32_t *kernel,
    int32_t *tensorOut,
    int N,
    int IH, int IW, int IC, int OC,
    int KH, int KW
)
{
    /* Code Starts Here */
    // printf("N: %d, IH: %d, IW: %d, IC: %d, OC: %d, KH: %d, KW: %d\n", N, IH, IW, IC, OC, KH, KW);

    _N = N;
    _IH = IH; 
    _IW = IW; 
    _IC = IC; 
    _OC = OC;
    _KH = KH; 
    _KW = KW;


    _kernel = (int32_t*)malloc(sizeof(int32_t) * OC * IC * KH * KW);
    for(int r = 0; r < OC; r++) {
        for(int c = 0; c < IC*KH*KW; c++) {
            _kernel[r*IC*KH*KW + c] = c_ker(r, c, kernel);
        }
    }
 
    _tensorIn = (int32_t*)malloc(sizeof(int32_t) * IC * KH * KW * IH * IW);
    for(int n = 0; n < N; n++) {
        for(int r = 0; r < IH*IW; r++) {
            for(int c = 0; c < IC*KH*KW; c++) {
                _tensorIn[r * IC*KH*KW + c] = c_in(n, c, r, tensorIn);
            }
        }

        // naive matmul
        for(int i = 0; i < IH*IW; i++) {
            for(int j = 0; j < OC; j++) {
                int32_t temp = 0;
                for(int k = 0; k < IC*KH*KW; k++) {
                    temp += _tensorIn[i * IC*KH*KW + k] * _kernel[j * IC*KH*KW + k];
                }
                tensorOut[n * IH*IW*OC + i * OC + j] = temp;
            }
        }
    }

    free(_tensorIn);
    free(_kernel);

    return 0;
    /* Code Ends Here */
}