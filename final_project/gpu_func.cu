#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"

__global__
void device_add_one(int* d_result, int t) {
    *d_result = t + 1;
}

/*
Just a dummy function that can be used to warm up GPU
*/
int useless_gpu_add_one(int t) {
    int result;
    int* d_result;

    checkCudaErrors(cudaMalloc((void**)&d_result, 1 * sizeof(int)));

    event_pair timer;
    start_timer(&timer);
    device_add_one<<<1,1>>>(d_result, t);
    check_launch("device_add_one");
    double time = stop_timer(&timer);

    std::cout << "device_add_one took: " << time << " seconds" << std::endl;

    checkCudaErrors(cudaMemcpy(&result, d_result, 1 * sizeof(int),
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_result));
    return result;
}


__global__
void device_gemm(double* __restrict__ A, double* __restrict__ B,
           double* __restrict__ C, double alpha, double beta,
           int M, int N, int K) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if ((ix < M) && (iy < N)) {
        printf("A: \n");
    printmat(A, M, K);
    printf("B: \n");
    printmat(B, K, N);
    printf("C: \n");
    printmat(C, M, N);
        for (int i=0; i<K; i++) {
            printf("C(%d,%d)=%f\n", ix, iy, C[ix+iy*M]);
            printf("A(%d,%d)=%f\n", ix, i, A[ix+i*M]);
            printf("B(%d,%d)=%f\n", i, iy, B[i+iy*K]);
            C[ix+iy*M]=alpha*A[ix+i*M]*B[i+iy*K]+beta*C[ix+iy*M];
        }
    }
}

void printmat(double* __restrict__ mat, int m, int n) {
    for (int i=0; i<M; i++) {
        for (int j=0; j< N; j++) {
            printf("%f ", mat(i+J*M));
        }
        printf("\n");
    }
}

/*
Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
*/
int myGEMM(double* __restrict__ A, double* __restrict__ B,
           double* __restrict__ C, double* alpha, double* beta,
           int M, int N, int K) {
    /* TODO: Write an efficient GEMM implementation on GPU */
    double al=*alpha;
    double be=*beta;
    printf("A: \n");
    printmat(A, M, K);
    printf("B: \n");
    printmat(B, K, N);
    printf("C: \n");
    printmat(C, M, N);
    int block_size_x = 32;
    int block_size_y = 32;
    int numBlocks_x = (N + block_size_x - 1) / block_size_x;
    int numBlocks_y = (M + block_size_y - 1) / (block_size_y);
    dim3 threads(block_size_x, block_size_y);
    dim3 blocks(numBlocks_x, numBlocks_y);
    device_gemm<<<blocks, threads>>>(A, B, C, al, be, M, N, K);
    printf("C: \n");
    printmat(C, M, N);
    return 0;
}
