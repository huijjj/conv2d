#include <time.h>

// for debugging
#include <string.h>
#include <stdio.h>

double benchmark(
    float *tensorIn,
    float *kernel,
    float *tensorOut,
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

int inference(
    float *tensorIn,
    float *kernel,
    float *tensorOut,
    int N,
    int IH, int IW, int IC, int OC,
    int KH, int KW
)
{
    
    // for debugging
    if(KW == 7) {
        printf("N: %d, IH: %d, IW: %d, IC: %d, OC: %d, KH: %d, KW: %d\n", N, IH, IW, IC, OC, KH, KW);
        for(int i = 0; i < N*IH*IW*IC; i++)
            tensorIn[i] = 1.0;
        for(int i = 0; i < OC*KH*KW*IC; i++)
            kernel[i] = 1.0;
    }



    for(int n = 0; n < N; n++) {
        for(int h = 0; h < IH; h++) {
            for(int w = 0; w < IW; w++) {
                for(int oc = 0; oc < OC; oc++) {
                    
                    int sh = h - (KH / 2);
                    int sw = w - (KW / 2);

                    float temp = 0;

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
                    if(KW == 7 && !oc) {
                        printf("%f ", temp);
                    }
                }

            }
            
            if(KW == 7) {
                return 0;
            }
        }
    }

    return 0;
    /* Code Ends Here */
}
