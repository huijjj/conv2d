#include <time.h>
#include <stdint.h>

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <memory.h>
#include <math.h>

#define NUMTHREAD 4
#define UNUSED -1

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
int32_t c_ker(int r, int c, int32_t* st) {
    return *(st + r * (_KH*_KW*_IC) + ((c % (_KH*_KW)) / _KW) * (_KW*_IC) + ((c % (_KH*_KW)) % _KW) * _IC + (c / (_KH*_KW)));
}

// converts input into matrix, gets element in row r and column c of converted matrix
int32_t c_in(int n, int r, int c, int32_t* st) {
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

typedef struct {
    int t;
    int n;
    int32_t** out;
} args;


void* img2col(void* arg) {
    int t = ((args*)arg)->t;
    int n = ((args*)arg)->n;
    int32_t* _out = (int32_t*)malloc(sizeof(int32_t) * (_IH * _IW / NUMTHREAD) * _IC * _KH * _KW);
    
    *(((args*)arg)->out) = _out;

    for(int r = 0; r < (_IH*_IW / NUMTHREAD); r++) {
        for(int c = 0; c < _IC*_KH*_KW; c++) {
            _out[r * _IC*_KH*_KW + c] = c_in(n, c, r + t * (_IH * _IW / NUMTHREAD), _tensorIn);
        }
    }

    pthread_exit(NULL);
}

void* ker2col(void* arg) {
    int t = ((args*)arg)->t;
    int32_t* _out = (int32_t*)malloc(sizeof(int32_t) * (_OC / NUMTHREAD) * _IC * _KH * _KW);

    *(((args*)arg)->out) = _out;

    for(int r = 0; r < (_OC / NUMTHREAD); r++) {
        for(int c = 0; c < _IC*_KH*_KW; c++) {
            _out[r*_IC*_KH*_KW + c] = c_ker(r + t * (_OC/NUMTHREAD), c, _kernel);
        }
    }

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
    // printf("N: %d, IH: %d, IW: %d, IC: %d, OC: %d, KH: %d, KW: %d\n", N, IH, IW, IC, OC, KH, KW);

    _N = N;
    _IH = IH; 
    _IW = IW; 
    _IC = IC; 
    _OC = OC;
    _KH = KH; 
    _KW = KW;
    _tensorIn = tensorIn;
    _kernel = kernel;

    int a_st;
    int b_st;
    int c_st;
    int d_st;
    pthread_t a;
    pthread_t b;
    pthread_t c;
    pthread_t d;
    int32_t* a_ker = NULL;
    int32_t* b_ker = NULL;
    int32_t* c_ker = NULL;
    int32_t* d_ker = NULL;
    args _a_arg = { 0, UNUSED, &a_ker };
    args _b_arg = { 1, UNUSED, &b_ker };
    args _c_arg = { 2, UNUSED, &c_ker };
    args _d_arg = { 3, UNUSED, &d_ker };
    pthread_create(&a, NULL, ker2col, &_a_arg);
    pthread_create(&b, NULL, ker2col, &_b_arg);
    pthread_create(&c, NULL, ker2col, &_c_arg);
    pthread_create(&d, NULL, ker2col, &_d_arg);
    pthread_join(a, &a_st);
    pthread_join(b, &b_st);
    pthread_join(c, &c_st);
    pthread_join(d, &d_st);
    int32_t* _ker[] = { a_ker, b_ker, c_ker, d_ker };

    for(int n = 0; n < N; n++) {
        int32_t* a_in = NULL;
        int32_t* b_in = NULL;
        int32_t* c_in = NULL;
        int32_t* d_in = NULL;
        args a_arg = { 0, n, &a_in };
        args b_arg = { 1, n, &b_in };
        args c_arg = { 2, n, &c_in };
        args d_arg = { 3, n, &d_in };
        pthread_create(&a, NULL, img2col, &a_arg);
        pthread_create(&b, NULL, img2col, &b_arg);
        pthread_create(&c, NULL, img2col, &c_arg);
        pthread_create(&d, NULL, img2col, &d_arg);
        pthread_join(a, &a_st);
        pthread_join(b, &b_st);
        pthread_join(c, &c_st);
        pthread_join(d, &d_st);
        int32_t* _in[] = { a_in, b_in, c_in, d_in };

        // naive matmul
        for(int i = 0; i < IH*IW; i++) {
            for(int j = 0; j < OC; j++) {
                int32_t temp = 0;
                for(int k = 0; k < IC*KH*KW; k++) {
                    temp += _in[i / (IH*IW/NUMTHREAD)][(i % (IH*IW/NUMTHREAD)) * IC*KH*KW + k] * _ker[j / (OC/NUMTHREAD)][(j % (OC/NUMTHREAD)) * IC*KH*KW + k];
                }
                tensorOut[n * IH*IW*OC + i * OC + j] = temp;
            }
        }


        free(a_in);
        free(b_in);
        free(c_in);
        free(d_in);
    }

    free(a_ker);
    free(b_ker);
    free(c_ker);
    free(d_ker);

    return 0;
    /* Code Ends Here */
}