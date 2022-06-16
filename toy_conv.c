#include <time.h>
#include <stdint.h>

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
int32_t c_ker(int r, int c, int32_t* st, int OC, int IC, int KH, int KW) {
    return *(st + r * KH*KW*IC + ((c % IC) / KW) * KW*IC + ((c % IC) % KH) * IC + (c / IC));
}

// converts input into matrix, gets element in row r and column c of converted matrix
int32_t c_in(int n, int r, int c, int32_t* st, int IH, int IW, int IC, int KH, int KW) {
    // element at (r , c) gets multiplied with (x, r) element in columnized kernel
    // and will be added up to element at (c / IH, c % IH, c), (H, W, C) at output

    // we want (h, w, ic) of original input
    // ic == r / (KH * KW)
    
    // first element should be ((c / IH) - (KH / 2), (c % IH) - (KW / 2))
    // r % IC is the index with in kernel
    const int h = (c / IH) - (KH / 2) + ((r % IC) / KH); 
    const int w = (c % IH) - (KW / 2) + ((r % IC) % KH);

    if(h < 0 || h >= IH || w < 0 || w >= IW) { // padding
        return 0;
    }
    else {
        return *(st + n * IH*IW*IC + h * IW*IC + w * IC + (r / (KH*KW)));
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
    for(int n = 0; n < N; n++) {
        for(int ih = 0; ih < IH; ih++) {
            for(int iw = 0; iw < IW; iw++) {
                for(int oc = 0; oc < OC; oc++) {

                    int32_t temp = 0;

                    for(int r = 0; r < KH*KW*IC; r++) {
                        for(int c = 0; c < IH*IW; c++) {
                            temp += c_in(n, r, c, tensorIn, IH, IW, IC, KH, KW) * c_ker(oc, r, kernel, OC, IC, KH, KW);
                        }
                    }

                    *(tensorOut + n * IH*IW*OC + ih * IW*OC + iw * OC + oc) = temp;
                }
            }
        }
    }




    return 0;
    /* Code Ends Here */
}