#include <time.h>
#include <stdint.h>

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <memory.h>

#define NUMTHREAD 4
#define UNUSED 0

int _N;
int _IH;
int _IW; 
int _IC; 
int _OC;
int _KH; 
int _KW;

int32_t* _tensorIn;
int32_t* _tensorOut;
int32_t* _kernel;

int inference(int32_t*, int32_t*, int32_t*, int, int, int, int, int, int, int);

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
    void** out;
    void** ker;
} args;


void* ker2col_144(void* arg) {
    int t = ((args*)arg)->t;
    uint8_t* _out = (uint8_t*)malloc(sizeof(uint8_t) * (_OC / NUMTHREAD) * 144);

    *(((args*)arg)->out) = _out;

    for(int r = 0; r < (_OC / NUMTHREAD); r++) {
        for(int c = 0; c < 144; c++) {
            *(_out) = c_ker(r + t * (_OC/NUMTHREAD), c, _kernel);
            _out = _out + 1;
        }
    }

    pthread_exit(NULL);
}
void* ker2col_288(void* arg) {
    int t = ((args*)arg)->t;
    uint8_t* _out = (uint8_t*)malloc(sizeof(uint8_t) * (_OC / NUMTHREAD) * 288);

    *(((args*)arg)->out) = _out;

    for(int r = 0; r < (_OC / NUMTHREAD); r++) {
        for(int c = 0; c < 288; c++) {
            *(_out) = c_ker(r + t * (_OC/NUMTHREAD), c, _kernel);
            _out = _out + 1;
        }
    }

    pthread_exit(NULL);
}
void* ker2col_576(void* arg) {
    int t = ((args*)arg)->t;
    uint8_t* _out = (uint8_t*)malloc(sizeof(uint8_t) * (_OC / NUMTHREAD) * 576);

    *(((args*)arg)->out) = _out;

    for(int r = 0; r < (_OC / NUMTHREAD); r++) {
        for(int c = 0; c < 576; c++) {
            *(_out) = c_ker(r + t * (_OC/NUMTHREAD), c, _kernel);
            _out = _out + 1;
        }
    }

    pthread_exit(NULL);
}
void* ker2col_800(void* arg) {
    int t = ((args*)arg)->t;
    uint8_t* _out = (uint8_t*)malloc(sizeof(uint8_t) * (_OC / NUMTHREAD) * 800);

    *(((args*)arg)->out) = _out;

    for(int r = 0; r < (_OC / NUMTHREAD); r++) {
        for(int c = 0; c < 800; c++) {
            *(_out) = c_ker(r + t * (_OC/NUMTHREAD), c, _kernel);
            _out = _out + 1;
        }
    }

    pthread_exit(NULL);
}
void* ker2col_1568(void* arg) {
    int t = ((args*)arg)->t;
    uint8_t* _out = (uint8_t*)malloc(sizeof(uint8_t) * (_OC / NUMTHREAD) * 1568);

    *(((args*)arg)->out) = _out;

    for(int r = 0; r < (_OC / NUMTHREAD); r++) {
        for(int c = 0; c < 1568; c++) {
            *(_out) = c_ker(r + t * (_OC/NUMTHREAD), c, _kernel);
            _out = _out + 1;
        }
    }

    pthread_exit(NULL);
}

void* img2col_matmul_144(void* arg) {
    int t = ((args*)arg)->t;
    register uint8_t** ker = ((args*)arg)->ker;

    register int32_t* _out = _tensorOut + t * (_N / NUMTHREAD) * _IH * _IW * _OC;
    uint8_t _img[_IH * _IW * 144 * _N / NUMTHREAD];
    register uint8_t* __img = _img;

    if(_IH*_IW == 256) {
        for(int n = 0; n < (_N / NUMTHREAD); n++) {
            for(int r = 0; r < 256; r++) {
                for(int c = 0; c < 144; c++) {
                    *(__img) = c_in(n + t * (_N/NUMTHREAD), c, r, _tensorIn);
                    __img = __img + 1;
                }
            }
        }

        register int32_t temp;
        for(int n = 0; n < _N / NUMTHREAD; n++) {
            for(int i = 0; i < 256; i++) {
                for(int j = 0; j < _OC; j++) {
                    temp = 0;
                    for(int k = 0; k < 144; k++) {
                        temp = temp + _img[n * 256*144 + i * 144 + k] * ker[j / (_OC / NUMTHREAD)][(j % (_OC / NUMTHREAD)) * 144 + k];
                    }
                    *(_out) = temp;
                    _out = _out + 1;
                }
            }
        }
    }
    else if(_IH*_IW == 1024) {
        for(int n = 0; n < (_N / NUMTHREAD); n++) {
            for(int r = 0; r < 1024; r++) {
                for(int c = 0; c < 144; c++) {
                    *(__img) = c_in(n + t * (_N/NUMTHREAD), c, r, _tensorIn);
                    __img = __img + 1;
                }
            }
        }

        register int32_t temp;
        for(int n = 0; n < _N / NUMTHREAD; n++) {
            for(int i = 0; i < 1024; i++) {
                for(int j = 0; j < _OC; j++) {
                    temp = 0;
                    for(int k = 0; k < 144; k++) {
                        temp = temp + _img[n * 1024*144 + i * 144 + k] * ker[j / (_OC / NUMTHREAD)][(j % (_OC / NUMTHREAD)) * 144 + k];
                    }
                    *(_out) = temp;
                    _out = _out + 1;
                }
            }
        }
    }
    else {
        for(int n = 0; n < (_N / NUMTHREAD); n++) {
            for(int r = 0; r < 4096; r++) {
                for(int c = 0; c < 144; c++) {
                    *(__img) = c_in(n + t * (_N/NUMTHREAD), c, r, _tensorIn);
                    __img = __img + 1;
                }
            }
        }

        register int32_t temp;
        for(int n = 0; n < _N / NUMTHREAD; n++) {
            for(int i = 0; i < 4096; i++) {
                for(int j = 0; j < _OC; j++) {
                    temp = 0;
                    for(int k = 0; k < 144; k++) {
                        temp = temp + _img[n * 4096*144 + i * 144 + k] * ker[j / (_OC / NUMTHREAD)][(j % (_OC / NUMTHREAD)) * 144 + k];
                    }
                    *(_out) = temp;
                    _out = _out + 1;
                }
            }
        }
    }
    
    pthread_exit(NULL);
}
void* img2col_matmul_288(void* arg) {
    int t = ((args*)arg)->t;
    register uint8_t** ker = ((args*)arg)->ker;

    register int32_t* _out = _tensorOut + t * (_N / NUMTHREAD) * _IH * _IW * _OC;
    uint8_t _img[_IH * _IW * 288 * _N / NUMTHREAD];
    register uint8_t* __img = _img;

    if(_IH*_IW == 256) {
        for(int n = 0; n < (_N / NUMTHREAD); n++) {
            for(int r = 0; r < 256; r++) {
                for(int c = 0; c < 288; c++) {
                    *(__img) = c_in(n + t * (_N/NUMTHREAD), c, r, _tensorIn);
                    __img = __img + 1;
                }
            }
        }

        register int32_t temp;
        for(int n = 0; n < _N / NUMTHREAD; n++) {
            for(int i = 0; i < 256; i++) {
                for(int j = 0; j < _OC; j++) {
                    temp = 0;
                    for(int k = 0; k < 288; k++) {
                        temp = temp + _img[n * 256*288 + i * 288 + k] * ker[j / (_OC / NUMTHREAD)][(j % (_OC / NUMTHREAD)) * 288 + k];
                    }
                    *(_out) = temp;
                    _out = _out + 1;
                }
            }
        }

    }
    else if(_IH*_IW == 1024) {
        for(int n = 0; n < (_N / NUMTHREAD); n++) {
            for(int r = 0; r < 1024; r++) {
                for(int c = 0; c < 288; c++) {
                    *(__img) = c_in(n + t * (_N/NUMTHREAD), c, r, _tensorIn);
                    __img = __img + 1;
                }
            }
        }

        register int32_t temp;
        for(int n = 0; n < _N / NUMTHREAD; n++) {
            for(int i = 0; i < 1024; i++) {
                for(int j = 0; j < _OC; j++) {
                    temp = 0;
                    for(int k = 0; k < 288; k++) {
                        temp = temp + _img[n * 1024*288 + i * 288 + k] * ker[j / (_OC / NUMTHREAD)][(j % (_OC / NUMTHREAD)) * 288 + k];
                    }
                    *(_out) = temp;
                    _out = _out + 1;
                }
            }
        }
        
    }
    else {
        for(int n = 0; n < (_N / NUMTHREAD); n++) {
            for(int r = 0; r < 4096; r++) {
                for(int c = 0; c < 288; c++) {
                    *(__img) = c_in(n + t * (_N/NUMTHREAD), c, r, _tensorIn);
                    __img = __img + 1;
                }
            }
        }

        register int32_t temp;
        for(int n = 0; n < _N / NUMTHREAD; n++) {
            for(int i = 0; i < 4096; i++) {
                for(int j = 0; j < _OC; j++) {
                    temp = 0;
                    for(int k = 0; k < 288; k++) {
                        temp = temp + _img[n * 4096*288 + i * 288 + k] * ker[j / (_OC / NUMTHREAD)][(j % (_OC / NUMTHREAD)) * 288 + k];
                    }
                    *(_out) = temp;
                    _out = _out + 1;
                }
            }
        }

    }

    
    pthread_exit(NULL);
}
void* img2col_matmul_576(void* arg) {
    int t = ((args*)arg)->t;
    register uint8_t** ker = ((args*)arg)->ker;

    register int32_t* _out = _tensorOut + t * (_N / NUMTHREAD) * _IH * _IW * _OC;
    uint8_t _img[_IH * _IW * 576 * _N / NUMTHREAD];
    register uint8_t* __img = _img;

    if(_IH*_IW == 256) {
        for(int n = 0; n < (_N / NUMTHREAD); n++) {
            for(int r = 0; r < 256; r++) {
                for(int c = 0; c < 576; c++) {
                    *(__img) = c_in(n + t * (_N/NUMTHREAD), c, r, _tensorIn);
                    __img = __img + 1;
                }
            }
        }

        register int32_t temp;
        for(int n = 0; n < _N / NUMTHREAD; n++) {
            for(int i = 0; i < 256; i++) {
                for(int j = 0; j < _OC; j++) {
                    temp = 0;
                    for(int k = 0; k < 576; k++) {
                        temp = temp + _img[n * 256*576 + i * 576 + k] * ker[j / (_OC / NUMTHREAD)][(j % (_OC / NUMTHREAD)) * 576 + k];
                    }
                    *(_out) = temp;
                    _out = _out + 1;
                }
            }
        }

    }
    else if(_IH*_IW == 1024) {
        for(int n = 0; n < (_N / NUMTHREAD); n++) {
            for(int r = 0; r < 1024; r++) {
                for(int c = 0; c < 576; c++) {
                    *(__img) = c_in(n + t * (_N/NUMTHREAD), c, r, _tensorIn);
                    __img = __img + 1;
                }
            }
        }

        register int32_t temp;
        for(int n = 0; n < _N / NUMTHREAD; n++) {
            for(int i = 0; i < 1024; i++) {
                for(int j = 0; j < _OC; j++) {
                    temp = 0;
                    for(int k = 0; k < 576; k++) {
                        temp = temp + _img[n * 1024*576 + i * 576 + k] * ker[j / (_OC / NUMTHREAD)][(j % (_OC / NUMTHREAD)) * 576 + k];
                    }
                    *(_out) = temp;
                    _out = _out + 1;
                }
            }
        }
        
    }
    else {
        for(int n = 0; n < (_N / NUMTHREAD); n++) {
            for(int r = 0; r < 4096; r++) {
                for(int c = 0; c < 576; c++) {
                    *(__img) = c_in(n + t * (_N/NUMTHREAD), c, r, _tensorIn);
                    __img = __img + 1;
                }
            }
        }

        register int32_t temp;
        for(int n = 0; n < _N / NUMTHREAD; n++) {
            for(int i = 0; i < 4096; i++) {
                for(int j = 0; j < _OC; j++) {
                    temp = 0;
                    for(int k = 0; k < 576; k++) {
                        temp = temp + _img[n * 4096*576 + i * 576 + k] * ker[j / (_OC / NUMTHREAD)][(j % (_OC / NUMTHREAD)) * 576 + k];
                    }
                    *(_out) = temp;
                    _out = _out + 1;
                }
            }
        }

    }
    pthread_exit(NULL);
}
void* img2col_matmul_800(void* arg) {
    int t = ((args*)arg)->t;
    register uint8_t** ker = ((args*)arg)->ker;

    register int32_t* _out = _tensorOut + t * (_N / NUMTHREAD) * _IH * _IW * _OC;
    uint8_t _img[_IH * _IW * 800 * _N / NUMTHREAD];
    register uint8_t* __img = _img;

    if(_IH*_IW == 256) {
        for(int n = 0; n < (_N / NUMTHREAD); n++) {
            for(int r = 0; r < 256; r++) {
                for(int c = 0; c < 800; c++) {
                    *(__img) = c_in(n + t * (_N/NUMTHREAD), c, r, _tensorIn);
                    __img = __img + 1;
                }
            }
        }

        register int32_t temp;
        for(int n = 0; n < _N / NUMTHREAD; n++) {
            for(int i = 0; i < 256; i++) {
                for(int j = 0; j < _OC; j++) {
                    temp = 0;
                    for(int k = 0; k < 800; k++) {
                        temp = temp + _img[n * 256*800 + i * 800 + k] * ker[j / (_OC / NUMTHREAD)][(j % (_OC / NUMTHREAD)) * 800 + k];
                    }
                    *(_out) = temp;
                    _out = _out + 1;
                }
            }
        }
    }
    else if(_IH*_IW == 1024) {
        for(int n = 0; n < (_N / NUMTHREAD); n++) {
            for(int r = 0; r < 1024; r++) {
                for(int c = 0; c < 800; c++) {
                    *(__img) = c_in(n + t * (_N/NUMTHREAD), c, r, _tensorIn);
                    __img = __img + 1;
                }
            }
        }

        register int32_t temp;
        for(int n = 0; n < _N / NUMTHREAD; n++) {
            for(int i = 0; i < 1024; i++) {
                for(int j = 0; j < _OC; j++) {
                    temp = 0;
                    for(int k = 0; k < 800; k++) {
                        temp = temp + _img[n * 1024*800 + i * 800 + k] * ker[j / (_OC / NUMTHREAD)][(j % (_OC / NUMTHREAD)) * 800 + k];
                    }
                    *(_out) = temp;
                    _out = _out + 1;
                }
            }
        }
    }
    else {
        for(int n = 0; n < (_N / NUMTHREAD); n++) {
            for(int r = 0; r < 4096; r++) {
                for(int c = 0; c < 800; c++) {
                    *(__img) = c_in(n + t * (_N/NUMTHREAD), c, r, _tensorIn);
                    __img = __img + 1;
                }
            }
        }

        register int32_t temp;
        for(int n = 0; n < _N / NUMTHREAD; n++) {
            for(int i = 0; i < 4096; i++) {
                for(int j = 0; j < _OC; j++) {
                    temp = 0;
                    for(int k = 0; k < 800; k++) {
                        temp = temp + _img[n * 4096*800 + i * 800 + k] * ker[j / (_OC / NUMTHREAD)][(j % (_OC / NUMTHREAD)) * 800 + k];
                    }
                    *(_out) = temp;
                    _out = _out + 1;
                }
            }
        }
    }

    pthread_exit(NULL);
}
void* img2col_matmul_1568(void* arg) {
    int t = ((args*)arg)->t;
    register uint8_t** ker = ((args*)arg)->ker;

    register int32_t* _out = _tensorOut + t * (_N / NUMTHREAD) * _IH * _IW * _OC;
    uint8_t _img[_IH * _IW * 1568 * _N / NUMTHREAD];
    register uint8_t* __img = _img;

    if(_IH*_IW == 256) {
        for(int n = 0; n < (_N / NUMTHREAD); n++) {
            for(int r = 0; r < 256; r++) {
                for(int c = 0; c < 1568; c++) {
                    *(__img) = c_in(n + t * (_N/NUMTHREAD), c, r, _tensorIn);
                    __img = __img + 1;
                }
            }
        }

        register int32_t temp;
        for(int n = 0; n < _N / NUMTHREAD; n++) {
            for(int i = 0; i < 256; i++) {
                for(int j = 0; j < _OC; j++) {
                    temp = 0;
                    for(int k = 0; k < 1568; k++) {
                        temp = temp + _img[n * 256*1568 + i * 1568 + k] * ker[j / (_OC / NUMTHREAD)][(j % (_OC / NUMTHREAD)) * 1568 + k];
                    }
                    *(_out) = temp;
                    _out = _out + 1;
                }
            }
        }

    }
    else if(_IH*_IW == 1024) {
        for(int n = 0; n < (_N / NUMTHREAD); n++) {
            for(int r = 0; r < 1024; r++) {
                for(int c = 0; c < 1568; c++) {
                    *(__img) = c_in(n + t * (_N/NUMTHREAD), c, r, _tensorIn);
                    __img = __img + 1;
                }
            }
        }

        register int32_t temp;
        for(int n = 0; n < _N / NUMTHREAD; n++) {
            for(int i = 0; i < 1024; i++) {
                for(int j = 0; j < _OC; j++) {
                    temp = 0;
                    for(int k = 0; k < 1568; k++) {
                        temp = temp + _img[n * 1024*1568 + i * 1568 + k] * ker[j / (_OC / NUMTHREAD)][(j % (_OC / NUMTHREAD)) * 1568 + k];
                    }
                    *(_out) = temp;
                    _out = _out + 1;
                }
            }
        }
        
    }
    else {
        for(int n = 0; n < (_N / NUMTHREAD); n++) {
            for(int r = 0; r < 4096; r++) {
                for(int c = 0; c < 1568; c++) {
                    *(__img) = c_in(n + t * (_N/NUMTHREAD), c, r, _tensorIn);
                    __img = __img + 1;
                }
            }
        }

        register int32_t temp;
        for(int n = 0; n < _N / NUMTHREAD; n++) {
            for(int i = 0; i < 4096; i++) {
                for(int j = 0; j < _OC; j++) {
                    temp = 0;
                    for(int k = 0; k < 1568; k++) {
                        temp = temp + _img[n * 4096*1568 + i * 1568 + k] * ker[j / (_OC / NUMTHREAD)][(j % (_OC / NUMTHREAD)) * 1568 + k];
                    }
                    *(_out) = temp;
                    _out = _out + 1;
                }
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
    _tensorOut = tensorOut;
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
    args _a_arg = { 0, &a_ker, UNUSED };
    args _b_arg = { 1, &b_ker, UNUSED };
    args _c_arg = { 2, &c_ker, UNUSED };
    args _d_arg = { 3, &d_ker, UNUSED };

    if(IC*KH*KW == 288) {
        pthread_create(&a, NULL, ker2col_288, &_a_arg);
        pthread_create(&b, NULL, ker2col_288, &_b_arg);
        pthread_create(&c, NULL, ker2col_288, &_c_arg);
        pthread_create(&d, NULL, ker2col_288, &_d_arg);
    }
    else if(IC*KH*KW == 144) {
        pthread_create(&a, NULL, ker2col_144, &_a_arg);
        pthread_create(&b, NULL, ker2col_144, &_b_arg);
        pthread_create(&c, NULL, ker2col_144, &_c_arg);
        pthread_create(&d, NULL, ker2col_144, &_d_arg);
    }
    else if(IC*KH*KW == 576) {
        pthread_create(&a, NULL, ker2col_576, &_a_arg);
        pthread_create(&b, NULL, ker2col_576, &_b_arg);
        pthread_create(&c, NULL, ker2col_576, &_c_arg);
        pthread_create(&d, NULL, ker2col_576, &_d_arg);
    }
    else if(IC*KH*KW == 800) {
        pthread_create(&a, NULL, ker2col_800, &_a_arg);
        pthread_create(&b, NULL, ker2col_800, &_b_arg);
        pthread_create(&c, NULL, ker2col_800, &_c_arg);
        pthread_create(&d, NULL, ker2col_800, &_d_arg);
    }
    else {
        pthread_create(&a, NULL, ker2col_1568, &_a_arg);
        pthread_create(&b, NULL, ker2col_1568, &_b_arg);
        pthread_create(&c, NULL, ker2col_1568, &_c_arg);
        pthread_create(&d, NULL, ker2col_1568, &_d_arg);
    }
    pthread_join(a, &a_st);
    pthread_join(b, &b_st);
    pthread_join(c, &c_st);
    pthread_join(d, &d_st);

    uint8_t* _ker[] = { a_ker, b_ker, c_ker, d_ker };
    args a_arg_ = { 0, UNUSED, _ker };
    args b_arg_ = { 1, UNUSED, _ker };
    args c_arg_ = { 2, UNUSED, _ker };
    args d_arg_ = { 3, UNUSED, _ker };
    if(IC*KH*KW == 288) {
        pthread_create(&a, NULL, img2col_matmul_288, &a_arg_);
        pthread_create(&b, NULL, img2col_matmul_288, &b_arg_);
        pthread_create(&c, NULL, img2col_matmul_288, &c_arg_);
        pthread_create(&d, NULL, img2col_matmul_288, &d_arg_);
    }
    else if(IC*KH*KW == 144) {
        pthread_create(&a, NULL, img2col_matmul_144, &a_arg_);
        pthread_create(&b, NULL, img2col_matmul_144, &b_arg_);
        pthread_create(&c, NULL, img2col_matmul_144, &c_arg_);
        pthread_create(&d, NULL, img2col_matmul_144, &d_arg_);
    }
    else if(IC*KH*KW == 576) {
        pthread_create(&a, NULL, img2col_matmul_576, &a_arg_);
        pthread_create(&b, NULL, img2col_matmul_576, &b_arg_);
        pthread_create(&c, NULL, img2col_matmul_576, &c_arg_);
        pthread_create(&d, NULL, img2col_matmul_576, &d_arg_);
    }
    else if(IC*KH*KW == 800) {
        pthread_create(&a, NULL, img2col_matmul_800, &a_arg_);
        pthread_create(&b, NULL, img2col_matmul_800, &b_arg_);
        pthread_create(&c, NULL, img2col_matmul_800, &c_arg_);
        pthread_create(&d, NULL, img2col_matmul_800, &d_arg_);
    }
    else {
        pthread_create(&a, NULL, img2col_matmul_1568, &a_arg_);
        pthread_create(&b, NULL, img2col_matmul_1568, &b_arg_);
        pthread_create(&c, NULL, img2col_matmul_1568, &c_arg_);
        pthread_create(&d, NULL, img2col_matmul_1568, &d_arg_);
    }

    pthread_join(a, &a_st);
    pthread_join(b, &b_st);
    pthread_join(c, &c_st);
    pthread_join(d, &d_st);

    free(a_ker);
    free(b_ker);
    free(c_ker);
    free(d_ker);

    return 0;
    /* Code Ends Here */
}
