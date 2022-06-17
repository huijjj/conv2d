#include <time.h>
#include <stdint.h>

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <memory.h>

#define NUM_THREAD 4

int32_t _kernel[4][8196];
int32_t * _tensorIn,
int _N;
int _IH;
int _IW; 
int _IC; 
int _OC;
int _KH; 
int _KW;


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
int32_t c_ker(int r, int c, int32_t* st, int IC, int KH, int KW) {
    return *(st + r * (KH*KW*IC) + ((c % (KH*KW)) / KW) * (KW*IC) + ((c % (KH*KW)) % KW) * IC + (c / (KH*KW)));
}

// converts input into matrix, gets element in row r and column c of converted matrix
int32_t c_in(int n, int r, int c, int32_t* st, int IH, int IW, int IC, int KH, int KW) {
    // element at (r , c) gets multiplied with (x, r) element in columnized kernel
    // and will be added up to element at (c / IH, c % IH, c), (H, W, C) at output

    // we want (h, w, ic) of original input
    // ic == r / (KH * KW)
    
    // first element should be ((c / IH) - (KH / 2), (c % IH) - (KW / 2))
    // r % IC is the index with in kernel
    const int h = (c / IW) - (KH / 2) + ((r % (KH*KW)) / KW); 
    const int w = (c % IW) - (KW / 2) + ((r % (KH*KW)) % KW);

    if(h < 0 || h >= IH || w < 0 || w >= IW) { // padding
        return 0;
    }
    else {
        return *(st + n * IH*IW*IC + h * IW*IC + w * IC + (r / (KH*KW)));
    }
}

void* foo(void* args) {






    pthread_exit(NULL);
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
    
    _N = N;
    _IH = IH; 
    _IW = IW; 
    _IC = IC; 
    _OC = OC;
    _KH = KH; 
    _KW = KW;
    _tensorIn = tensorIn;

    // split kernel into NUM_THREAD for multithreading
    for(int oc = 0; oc < OC; oc++) {   
        memcpy(_kernel[oc / (OC / NUM_THREAD)], kernel + oc*KH*KW*IC, sizeof(int32_t)*KH*KW*IC);
    }

    for(int n = 0; n < N; n++) {









    }

    return 0;
    /* Code Ends Here */
}