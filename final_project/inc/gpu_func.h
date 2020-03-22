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

int myGEMM(double* A, double* B, double* C, double* alpha, double* beta, int M, int N, int K);

void my_feedforward(NeuralNetwork& nn, const arma::mat& X, struct cache& cache, 
const arma::mat& b0r, const arma::mat& b1r, const arma::mat& T, double* a0, double* a1, double* z0, double* z1, double* yc, double* W1_test, double* W0_test, double* W3_test);
void my_backprop(NeuralNetwork& nn, const arma::mat& y, double reg, const struct cache& bpcache, struct grads& bpgrads);


#endif
