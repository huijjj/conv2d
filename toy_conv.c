#include <time.h>

double benchmark(
    float *tensorIn,
    float *kernel,
    float *tensorOut,
    int N,
    int IH, int IW, int IC, int OC,
    int KH, int KW
)
{
    int num_iter = 1; // was 500 originally

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
float c_ker(int r, int c, float* st, int OC, int IC, int KH, int KW) {
    return *(st + r * KH*KW*IC + ((c % IC) / KW) * KW*IC + ((c % IC) % KH) * IC + (c / IC));
}

// converts input into matrix, gets element in row r and column c of converted matrix
float c_in(int r, int c, float* st, int IH, int IW, int IC, int KH, int KW) {
    // element at (r , c) gets multiplied with (x, r) element in columnized kernel
    // and will be added up to element at (c / IH, c % IH, c), (H, W, C) at output

    // ic == r / (KH * KW)
    int c_r;
    int c_c;


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
    // printf("N: %d, IH: %d, IW: %d, IC: %d, OC: %d, KH: %d, KW: %d\n", N, IH, IW, IC, OC, KH, KW);

    unsigned int e = 0;
    for(int n = 0; n < N; n++) {



    }




    // for(int n = 0; n < N; n++) {
    //     for(int h = 0; h < IH; h++) {
    //         for(int w = 0; w < IW; w++) {
    //             for(int oc = 0; oc < OC; oc++) {
                    
    //                 int sh = h - (KH / 2);
    //                 int sw = w - (KW / 2);

    //                 float temp = 0;

    //                 for(int ic = 0; ic < IC; ic++) {
    //                     for(int kh = 0; kh < KH; kh++) {
    //                         for(int kw = 0; kw < KW; kw++) {

    //                             if(
    //                                 (sh + kh) >= 0 && 
    //                                 (sh + kh) < IH &&
    //                                 (sw + kw) >= 0 &&
    //                                 (sw + kw) < IW) {
    //                                 temp = temp + 
    //                                     *(tensorIn + n * IH*IW*IC + (sh + kh) * IW*IC + (sw + kw) * IC + ic) * 
    //                                     *(kernel + oc * KH*KW*IC + kh * KW*IC + kw * IC + ic);
    //                             }
    //                         }
    //                     }
    //                 }
                    
    //                 *(tensorOut + n * IH*IW*OC + h * IW*OC + w * OC + oc) = temp;
    //             }
    //         }
    //     }
    // }

    return 0;
    /* Code Ends Here */
}
