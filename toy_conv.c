#include <time.h>
#include <stdint.h>
#include <pthread.h>

// for debugging
#include <string.h>
#include <stdio.h>

double benchmark(
    int32_t *tensorIn,
    int32_t *kernel,
    int32_t *tensorOut,
    int N,
    int IH, int IW, int IC, int OC,
    int KH, int KW
)
{
    int num_iter = 1;

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

typedef struct _conv_arg {
    int32_t* dst;
    int32_t* in;
    int32_t* ker; 
    int n;
    int sh; 
    int sw; 
    int oc; 
    int KH; 
    int KW; 
    int IC; 
    int IH; 
    int IW;
} conv_arg;

void *conv(void *args) {
    int32_t* dst = ((conv_arg*)args)->dst;
    int32_t* in = ((conv_arg*)args)->in;
    int32_t* ker = ((conv_arg*)args)->ker;
    int n = ((conv_arg*)args)->n; 
    int sh = ((conv_arg*)args)->sh; 
    int sw = ((conv_arg*)args)->sw; 
    int oc = ((conv_arg*)args)->oc; 
    int KH = ((conv_arg*)args)->KH; 
    int KW = ((conv_arg*)args)->KW; 
    int IC = ((conv_arg*)args)->IC; 
    int IH = ((conv_arg*)args)->IH; 
    int IW = ((conv_arg*)args)->IW;

    pthread_t tid = pthread_self();

    int32_t temp = 0;
    for(int kh = 0; kh < KH; kh++) {
        for(int kw = 0; kw < KW; kw++) {
            for(int ic = 0; ic < IC; ic++) {
                if(
                    (sh + kh) >= 0 && 
                    (sh + kh) < IH &&
                    (sw + kw) >= 0 &&
                    (sw + kw) < IW) {
                    temp = temp + 
                        *(in + n * IH*IW*IC + (sh + kh) * IW*IC + (sw + kw) * IC + ic) * 
                        *(ker + oc * KH*KW*IC + kh * KW*IC + kw * IC + ic);
                }
            }
        }
    }
    
    *(dst) = temp;
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
    pthread_t thread[4];
    int num_thread;
    if(OC >= 4) {
        num_thread = 4;
    }
    else if(OC >= 2) {
        num_thread = 2;
    }
    else {
        num_thread = 1;
    }


    for(int n = 0; n < N; n++) {
        for(int h = 0; h < IH; h++) {
            for(int w = 0; w < IW; w++) {
                int oc;
                int sh = h - (KH / 2);
                int sw = w - (KW / 2);
                for(oc = 0; oc < OC; oc += num_thread) { // parallel programming
                    conv_arg args[4];
                    int32_t res[4];
                    for(int t = 0; t < num_thread; t++) {
                        args[t].dst = res + t;
                        args[t].in = tensorIn;
                        args[t].ker = kernel;
                        args[t].n = n;
                        args[t].sh = sh;
                        args[t].sw = sw;
                        args[t].oc = oc + t;
                        args[t].KH = KH;
                        args[t].KW = KW;
                        args[t].IC = IC;
                        args[t].IH = IH;
                        args[t].IW = IW;
                    }

                    for(int t = 0; t < num_thread; t++) {
                        if(pthread_create(&thread[t], NULL, conv, args + t)) {
                            return -1;
                        }
                    }

                    int status[4];
                    for(int t = 0; t < num_thread; t++) {
                        pthread_join(thread[t], (void **)&(status[4]));
                        *(tensorOut + n * IH*IW*OC + h * IW*OC + w * OC + oc + t) = res[t];
                    }
                }

                if(oc != OC) {
                    oc -= num_thread;

                    for(; oc < OC; oc++) {
                        int32_t temp = 0;

                        for(int kh = 0; kh < KH; kh++) {
                            for(int kw = 0; kw < KW; kw++) {
                                for(int ic = 0; ic < IC; ic++) {
                                    if(
                                        (sh + kh) >= 0 && 
                                        (sh + kh) < IH &&
                                        (sw + kw) >= 0 &&
                                        (sw + kw) < IW) {
                                        temp = temp + 
                                            *(tensorIn + n * IH*IW*IC + (sh + kh) * IW*IC + (sw + kw) * IC + ic) * 
                                            *(kernel + oc * KH*KW*IC + kh * KW*IC + kw * IC + ic);
                                    }
                                }
                            }
                        }
                        
                        *(tensorOut + n * IH*IW*OC + h * IW*OC + w * OC + oc) = temp;
                    }
                }
            }
        }
    }

    return 0;
    /* Code Ends Here */
}
