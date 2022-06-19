#include <time.h>
#include <stdint.h>

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <memory.h>

#define NUM_THREAD 4

int32_t* _tensorIn;
int32_t* _kernel;

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

typedef struct {
    int tid ;
    int n;
    int32_t** out;
} args;

void* conv(void* arg) {
    int t = ((args *)arg)->tid;
    int n = ((args *)arg)->n;    
    int32_t* _out = (int32_t*)malloc(sizeof(int32_t)*_IW*_IH*(_OC/NUM_THREAD));
    *(((args *)arg)->out) = _out;

    for(int h = 0; h < _IH; h++) {
        for(int w = 0; w < _IW; w++) {
            int sh = h - (_KH / 2);
            int sw = w - (_KW / 2);

            for(int oc = 0; oc < (_OC / NUM_THREAD); oc++) {

                int32_t temp = 0;
                for(int kh = 0; kh < _KH; kh++) {
                    for(int kw = 0; kw < _KW; kw++) {
                        for(int ic = 0; ic < _IC; ic++) {
                            if(
                                (sh + kh) >= 0 && 
                                (sh + kh) < _IH &&
                                (sw + kw) >= 0 &&
                                (sw + kw) < _IW) {
                                temp = temp + 
                                    *(_tensorIn + n * _IH*_IW*_IC + (sh + kh) * _IW*_IC + (sw + kw) * _IC + ic) * 
                                    _kernel[(oc + t * (_OC/NUM_THREAD)) * _KH*_KW*_IC + kh * _KW*_IC + kw * _IC + ic];
                            }
                        }
                    }
                }
                
                *(_out + h * _IW*(_OC / NUM_THREAD) + w * (_OC / NUM_THREAD) + oc) = temp;
            }
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

    for(int n = 0; n < N; n++) {
        int a_st;
        int b_st;
        int c_st;
        int d_st;
        pthread_t a;
        pthread_t b;
        pthread_t c;
        pthread_t d;
        int32_t* a_out = NULL;
        int32_t* b_out = NULL;
        int32_t* c_out = NULL;
        int32_t* d_out = NULL;
        args a_arg = { 0, n, &a_out };
        args b_arg = { 1, n, &b_out };
        args c_arg = { 2, n, &c_out };
        args d_arg = { 3, n, &d_out };
        pthread_create(&a, NULL, conv, &a_arg);
        pthread_create(&b, NULL, conv, &b_arg);
        pthread_create(&c, NULL, conv, &c_arg);
        pthread_create(&d, NULL, conv, &d_arg);
        pthread_join(a, &a_st);
        pthread_join(b, &b_st);
        pthread_join(c, &c_st);
        pthread_join(d, &d_st);

        // merge result
        for(int h = 0; h < IH; h++) {
            for(int w = 0; w < IW; w++) {
                for(int oc = 0; oc < OC; oc++) {
                    if(oc < (OC / NUM_THREAD)) {
                        *(tensorOut + n * IH*IW*OC + h * IW*OC + w * OC + oc) = *(a_out + h * IW*(OC / NUM_THREAD) + w*(OC / NUM_THREAD) + oc);    
                    }
                    else if(oc < (2 * (OC / NUM_THREAD))) {
                        *(tensorOut + n * IH*IW*OC + h * IW*OC + w * OC + oc) = *(b_out + h * IW*(OC / NUM_THREAD) + w*(OC / NUM_THREAD) + oc % (OC / NUM_THREAD));    
                    }
                    else if(oc < (3 * (OC / NUM_THREAD))) {
                        *(tensorOut + n * IH*IW*OC + h * IW*OC + w * OC + oc) = *(c_out + h * IW*(OC / NUM_THREAD) + w*(OC / NUM_THREAD) + oc % (OC / NUM_THREAD));    
                    }
                    else {
                        *(tensorOut + n * IH*IW*OC + h * IW*OC + w * OC + oc) = *(d_out + h * IW*(OC / NUM_THREAD) + w*(OC / NUM_THREAD) + oc % (OC / NUM_THREAD));    
                    }
                }
            }
        }

        free(a_out);
        free(b_out);
        free(c_out);
        free(d_out);
    }


    return 0;
    /* Code Ends Here */
}