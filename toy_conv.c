#include <time.h>
#include <stdint.h>

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <memory.h>
#include <math.h>
#include <arm_neon.h>


#define NUMTHREAD 4
#define UNUSED NULL

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
    int num_iter = 500; // originally 500

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
inline uint8_t c_ker(int r, int c, int32_t* st) {
    return (uint8_t)(*(st + r * (_KH*_KW*_IC) + ((c % (_KH*_KW)) / _KW) * (_KW*_IC) + ((c % (_KH*_KW)) % _KW) * _IC + (c / (_KH*_KW))));
}

// converts input into matrix, gets element in row r and column c of converted matrix
inline uint8_t c_in(int n, int r, int c, int32_t* st) {
    // element at (r , c) gets multiplied with (x, r) element in columnized kernel
    // and will be added up to element at (c / IH, c % IH, c), (H, W, C) at output

    // we want (h, w, ic) of original input
    // ic == r / (KH * KW)
    
    // first element should be ((c / IH) - (KH / 2), (c % IH) - (KW / 2))
    // r % IC is the index with in kernel
    const int h = (c / _IW) - (_KH / 2) + ((r % (_KH*_KW)) / _KW); 
    const int w = (c % _IW) - (_KW / 2) + ((r % (_KH*_KW)) % _KW);


    return (h < 0 || h >= _IH || w < 0 || w >= _IW) ? 0 : (uint8_t)(*(st + n * _IH*_IW*_IC + h * _IW*_IC + w * _IC + (r / (_KH*_KW))));
}

typedef struct {
    int t;
    int n;
    void** out;
    void** in;
    void* ker;
} args;


void* img2col(void* arg) {
    int t = ((args*)arg)->t;
    int n = ((args*)arg)->n;
    uint8_t* _out = (uint8_t*)malloc(sizeof(uint8_t) * (_IH * _IW / NUMTHREAD) * _IC * _KH * _KW);
    
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
    uint8_t* _out = (uint8_t*)malloc(sizeof(uint8_t) * (_OC / NUMTHREAD) * _IC * _KH * _KW);

    *(((args*)arg)->out) = _out;

    for(int r = 0; r < (_OC / NUMTHREAD); r++) {
        for(int c = 0; c < _IC*_KH*_KW; c++) {
            _out[r*_IC*_KH*_KW + c] = c_ker(r + t * (_OC/NUMTHREAD), c, _kernel);
        }
    }

    pthread_exit(NULL);
}

/*
 * Used in Coppersmith-Winograd Algorithm
 * strideA is the col num of matA, initial value is K
 * strideB is the col num of matB, initial value is N
 * strideC is the col num of matC, initial value is N
 */
inline static void mm_generate(int32_t* matA, int32_t* matB, int32_t* matC, const int M, const int N, const int K,
                        const int strideA, const int strideB, const int strideC){
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            int32_t sum = 0.0f;
            for(int k = 0; k < K; k++){
                sum += matA[i*strideA + k] * matB[k*strideB + j];
            }
            matC[i*strideC + j] = sum;
        }
    }
}

/*
 * matA M*K
 * matB K*N
 * matC M*N
 * matC = matA * matB
 * S1 = A21 + A22     T1 = B12 - B11
 * S2 = S1 - A11      T2 = B22 - T1
 * S3 = A11 - A21     T3 = B22 - B12
 * S4 = A12 - S2      T4 = T2 - B21
 * M1 = A11 * B11     U1 = M1 + M2
 * M2 = A12 * B21     U2 = M1 + M6
 * M3 = S4 * B22      U3 = U2 + M7
 * M4 = A22 * T4      U4 = U2 + M5
 * M5 = S1 * T1       U5 = U4 + M3
 * M6 = S2 * T2       U6 = U3 - U4
 * M7 = S3 * T3       U7 = U3 + M5
 * C11 = U1
 * C12 = U5
 * C21 = U6
 * C22 = U7
 */
inline static void mm_CoppersmithWinograd(float* matA, float* matB, float* matC, const int M, const int N, const int K,
                             const int strideA, const int strideB, const int strideC){
    if((M <= 2) || (M%2 != 0 || N%2 != 0 || K%2 != 0)){
        return mm_generate(matA, matB, matC, M, N, K, strideA, strideB, strideC);
    }

    float* S1 = (float*) malloc((M/2) * (K/2) * sizeof(float));
    float* S2 = (float*) malloc((M/2) * (K/2) * sizeof(float));
    float* S3 = (float*) malloc((M/2) * (K/2) * sizeof(float));
    float* S4 = (float*) malloc((M/2) * (K/2) * sizeof(float));
    {
        for(int i = 0; i < M/2; i++){
            for(int j = 0; j < K/2; j++){
                int idxA, offset, idxS = i * (K/2) + j;

                //S1     = A21 + A22
                idxA     = (i + (M/2)) * strideA + j;
                offset   = K/2;
                S1[idxS] = matA[idxA] + matA[idxA + offset];

                //S2     = S1 - A11
                idxA     = i * strideA + j;
                S2[idxS] = S1[idxS] - matA[idxA];

                //S3     = A11 - A21
                offset   = (M/2) * strideA;
                S3[idxS] = matA[idxA] - matA[idxA + offset];

                //S4     = A12 - S2
                idxA     = i * strideA + (K/2) + j;
                S4[idxS] = matA[idxA] - S2[idxS];
            }
        }
    }

    float* T1 = (float*) malloc((K/2) * (N/2) * sizeof(float));
    float* T2 = (float*) malloc((K/2) * (N/2) * sizeof(float));
    float* T3 = (float*) malloc((K/2) * (N/2) * sizeof(float));
    float* T4 = (float*) malloc((K/2) * (N/2) * sizeof(float));
    {
        for(int i = 0; i < K/2; i++){
            for(int j = 0; j < N/2; j++){
                int idxB, offset, idxT = i * (N/2) + j;

                //T1     = B12 - B11
                idxB     = i * strideB + j;
                offset   = (N/2);
                T1[idxT] = matB[idxB + offset] - matB[idxB];

                //T2     = B22 - T1
                idxB     = (i + (K/2)) * strideB + (N/2) + j;
                T2[idxT] = matB[idxB] - T1[idxT];

                //T3     = B22 - B12
                idxB     = i * strideB + (N/2) + j;
                offset   = ((K/2)) * strideB;
                T3[idxT] = matB[idxB + offset] - matB[idxB];

                //T4     = T2 - B21
                idxB     = (i + (K/2)) * strideB + j;
                T4[idxT] = T2[idxT] - matB[idxB];
            }
        }
    }

    //M1 = A11 * B11
    float* M1 = (float*) malloc((M/2) * (N/2) * sizeof(float));
    mm_CoppersmithWinograd(matA, matB, &M1[0], M/2, N/2, K/2, strideA, strideB, N/2);

    //M2 = A12 * B21
    float* M2 = (float*) malloc((M/2) * (N/2) * sizeof(float));
    mm_CoppersmithWinograd(&matA[K/2], &matB[(K/2)*strideB], &M2[0], M/2, N/2, K/2, strideA, strideB, N/2);

    //M3 = S4 * B22
    float* M3 = (float*) malloc((M/2) * (N/2) * sizeof(float));
    mm_CoppersmithWinograd(&S4[0], &matB[(K/2) * strideB + (N/2)], &M3[0], M/2, N/2, K/2, K/2, strideB, N/2);

    //M4 = A22 * T4
    float* M4 = (float*) malloc((M/2) * (N/2) * sizeof(float));
    mm_CoppersmithWinograd(&matA[(M/2) * strideA + (K/2)], &T4[0], &M4[0], M/2, N/2, K/2, strideA, N/2, N/2);

    //M5 = S1 * T1
    float* M5 = (float*) malloc((M/2) * (N/2) * sizeof(float));
    mm_CoppersmithWinograd(&S1[0], &T1[0], &M5[0], M/2, N/2, K/2, K/2, N/2, N/2);

    //M6 = S2 * T2
    float* M6 = (float*) malloc((M/2) * (N/2) * sizeof(float));
    mm_CoppersmithWinograd(&S2[0], &T2[0], &M6[0], M/2, N/2, K/2, K/2, N/2, N/2);

    //M7 = S3 * T3
    float* M7 = (float*) malloc((M/2) * (N/2) * sizeof(float));
    mm_CoppersmithWinograd(&S3[0], &T3[0], &M7[0], M/2, N/2, K/2, K/2, N/2, N/2);

    //C11 = U1 = M1 + M2
    //C12 = U5 = U4 + M3 = U2 + M5 + M3 = M1 + M6 + M5 + M3
    //C21 = U6 = U3 - M4 = U2 + M7 - M4 = M1 + M6 + M7 - M4
    //C22 = U7 = U3 + M5 = U2 + M7 + M5 = M1 + M6 + M7 + M5
    for(int i = 0; i < M/2; i++){
        for(int j = 0; j < N/2; j++){
            int idx = i * (N/2) + j;
            matC[i*strideC + j] = M1[idx] + M2[idx];
            matC[i*strideC + j + (N/2)] = M1[idx] + M6[idx] + M5[idx] + M3[idx];
            matC[(i+(M/2))*strideC + j] = M1[idx] + M6[idx] + M7[idx] - M4[idx];
            matC[(i+(M/2))*strideC + j + (N/2)] = M1[idx] + M6[idx] + M7[idx] + M5[idx];
        }
    }
    free(S1);           S1=NULL;
    free(S2);           S2=NULL;
    free(S3);           S3=NULL;
    free(S4);           S4=NULL;
    free(T1);           T1=NULL;
    free(T2);           T2=NULL;
    free(T3);           T3=NULL;
    free(T4);           T4=NULL;
    free(M1);           M1=NULL;
    free(M2);           M2=NULL;
    free(M3);           M3=NULL;
    free(M4);           M4=NULL;
    free(M5);           M5=NULL;
    free(M6);           M6=NULL;
    free(M7);           M7=NULL;
}

inline static void _mm_CoppersmithWinograd(float* matA, float* matB, float* matC, const int M, const int N, const int K){
    mm_CoppersmithWinograd(matA, matB, matC, M, N, K, K, N, N);
}


void* matmul_naive(void* arg) {
    int t = ((args*)arg)->t;
    uint8_t** in = ((args*)arg)->in;
    uint8_t* ker = ((args*)arg)->ker;
    int32_t* _out = (int32_t*)malloc(sizeof(int32_t) * (_OC / NUMTHREAD) * _IH * _IW);
    // int32_t* temp = (int32_t*)malloc(sizeof(int32_t) * (_OC/NUMTHREAD));

    *(((args*)arg)->out) = _out;

    for(int i = 0; i < _IH*_IW; i++) {
        // memset(temp, 0, sizeof(int32_t) * (_OC/NUMTHREAD));
        for(int j = 0; j < (_OC/NUMTHREAD); j++) {
            int32_t temp = 0;
            for(int k = 0; k < _IC*_KH*_KW; k++) {
                temp += in[i / (_IH*_IW/NUMTHREAD)][(i % (_IH*_IW/NUMTHREAD)) * _IC*_KH*_KW + k] * ker[j * _IC*_KH*_KW + k];
            }
            _out[i * (_OC/NUMTHREAD) + j] = temp;
        }

        // memcpy(_out + i * (_OC/NUMTHREAD), temp, sizeof(int32_t) * (_OC/NUMTHREAD));
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
    uint8_t* a_ker = NULL;
    uint8_t* b_ker = NULL;
    uint8_t* c_ker = NULL;
    uint8_t* d_ker = NULL;
    args _a_arg = { 0, UNUSED, &a_ker, UNUSED, UNUSED };
    args _b_arg = { 1, UNUSED, &b_ker, UNUSED, UNUSED };
    args _c_arg = { 2, UNUSED, &c_ker, UNUSED, UNUSED };
    args _d_arg = { 3, UNUSED, &d_ker, UNUSED, UNUSED };
    pthread_create(&a, NULL, ker2col, &_a_arg);
    pthread_create(&b, NULL, ker2col, &_b_arg);
    pthread_create(&c, NULL, ker2col, &_c_arg);
    pthread_create(&d, NULL, ker2col, &_d_arg);
    pthread_join(a, &a_st);
    pthread_join(b, &b_st);
    pthread_join(c, &c_st);
    pthread_join(d, &d_st);

    for(int n = 0; n < N; n++) {
        uint8_t* a_in = NULL;
        uint8_t* b_in = NULL;
        uint8_t* c_in = NULL;
        uint8_t* d_in = NULL;
        args a_arg = { 0, n, &a_in, UNUSED, UNUSED };
        args b_arg = { 1, n, &b_in, UNUSED, UNUSED };
        args c_arg = { 2, n, &c_in, UNUSED, UNUSED };
        args d_arg = { 3, n, &d_in, UNUSED, UNUSED };
        pthread_create(&a, NULL, img2col, &a_arg);
        pthread_create(&b, NULL, img2col, &b_arg);
        pthread_create(&c, NULL, img2col, &c_arg);
        pthread_create(&d, NULL, img2col, &d_arg);
        pthread_join(a, &a_st);
        pthread_join(b, &b_st);
        pthread_join(c, &c_st);
        pthread_join(d, &d_st);

        uint8_t* _in[] = { a_in, b_in, c_in, d_in };
        int32_t* a_out = NULL;
        int32_t* b_out = NULL;
        int32_t* c_out = NULL;
        int32_t* d_out = NULL;
        args a_arg_ = { 0, UNUSED, &a_out, _in, a_ker };
        args b_arg_ = { 1, UNUSED, &b_out, _in, b_ker };
        args c_arg_ = { 2, UNUSED, &c_out, _in, c_ker };
        args d_arg_ = { 3, UNUSED, &d_out, _in, d_ker };
        pthread_create(&a, NULL, matmul_naive, &a_arg_);
        pthread_create(&b, NULL, matmul_naive, &b_arg_);
        pthread_create(&c, NULL, matmul_naive, &c_arg_);
        pthread_create(&d, NULL, matmul_naive, &d_arg_);
        pthread_join(a, &a_st);
        pthread_join(b, &b_st);
        pthread_join(c, &c_st);
        pthread_join(d, &d_st);


        // merge output
        for(int h = 0; h <IH; h++) {
            for(int w = 0; w < IW; w++) {
                memcpy(tensorOut + n * IH*IW*OC + h * IW*OC + w * OC, a_out + h * IW*(OC/NUMTHREAD) + w * (OC/NUMTHREAD), sizeof(int32_t) * (OC/NUMTHREAD));
                memcpy(tensorOut + n * IH*IW*OC + h * IW*OC + w * OC + 1 * (OC / NUMTHREAD), b_out + h * IW*(OC/NUMTHREAD) + w * (OC/NUMTHREAD), sizeof(int32_t) * (OC/NUMTHREAD));
                memcpy(tensorOut + n * IH*IW*OC + h * IW*OC + w * OC + 2 * (OC / NUMTHREAD), c_out + h * IW*(OC/NUMTHREAD) + w * (OC/NUMTHREAD), sizeof(int32_t) * (OC/NUMTHREAD));
                memcpy(tensorOut + n * IH*IW*OC + h * IW*OC + w * OC + 3 * (OC / NUMTHREAD), d_out + h * IW*(OC/NUMTHREAD) + w * (OC/NUMTHREAD), sizeof(int32_t) * (OC/NUMTHREAD));
            }
        }

        free(a_in);
        free(b_in);
        free(c_in);
        free(d_in);
        free(a_out);
        free(b_out);
        free(c_out);
        free(d_out);
    }

    free(a_ker);
    free(b_ker);
    free(c_ker);
    free(d_ker);

    return 0;
    /* Code Ends Here */
}