#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "neural_network.h"
#include <armadillo>
#include "../utils/common.h"
struct event_pair {
    cudaEvent_t start;
    cudaEvent_t end;
};

inline void check_launch(const char* kernel_name) {
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();

    if(err != cudaSuccess) {
        std::cerr << "error in " << kernel_name << " kernel" << std::endl;
        std::cerr << "error was: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

inline void start_timer(event_pair* p) {
    cudaEventCreate(&p->start);
    cudaEventCreate(&p->end);
    cudaEventRecord(p->start, 0);
}


inline double stop_timer(event_pair* p) {
    cudaEventRecord(p->end, 0);
    cudaEventSynchronize(p->end);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, p->start, p->end);
    cudaEventDestroy(p->start);
    cudaEventDestroy(p->end);
    return elapsed_time;
}

int useless_gpu_add_one(int t);

int  myGEMM(double* __restrict__ A, double* __restrict__ B, double* __restrict__ C, double* alpha, double* beta, int M, int N, int K, bool A_T=false, bool B_T=false);
void gpu_addmat(double* dA, double* dB, double* dC, double alpha, double beta, int M, int N);
void gpu_sumcol(double* ddata, double* dresult, int M, int N) ;
void gpu_transpose(double* ddata, double* dresult, int M, int N);
void gpu_sigmoid(double* ddata, double* dresult, int M, int N);
void gpu_exp(double* ddata, double* dresult, int M, int N);
void gpu_softmax(double* ddata, double* dresult, int M, int N);
void gpu_sumrow(double* ddata, double* dresult, int M, int N);
void gpu_hadmard(double* c, double* a, double* b, int M, int N);
#endif
