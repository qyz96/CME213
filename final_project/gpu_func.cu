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
    for (int i=0; i<K; i++) {
        C[ix*N+iy]+=A[ix*K+i]*B[i*N+iy];
    }

}

/*
Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
*/
int myGEMM(double* __restrict__ A, double* __restrict__ B,
           double* __restrict__ C, double* alpha, double* beta,
           int M, int N, int K) {
    /* TODO: Write an efficient GEMM implementation on GPU */


    double *array_a  = nullptr;
    double *array_b = nullptr;
    double *array_c = nullptr;
    double al=*alpha;
    double be=*beta;

    // TODO: allocate GPU memory
    cudaMalloc(&array_a,  M*K*sizeof(double));
    cudaMalloc(&array_b, K*N*sizeof(double));
    cudaMalloc(&array_c, M*N*sizeof(double));
    //cudaMemset(device_input_array + num_nodes, 0, num_bytes_alloc - num_nodes);
    
    // TODO: check for allocation failure
    if (!array_a || !array_b || !array_c) 
     {
         std::cerr << "Couldn't allocate memory!" << std::endl;
         return 1;
     }
    // TODO: copy data to the GPU
    {
         cudaMemcpy(array_a, A, sizeof(double)*M*K, cudaMemcpyHostToDevice);
         cudaMemcpy(array_b, B, sizeof(double)*K*N, cudaMemcpyHostToDevice);
         cudaMemcpy(array_c, C, sizeof(double)*M*N, cudaMemcpyHostToDevice);
         check_launch("copy to gpu");

     }


    event_pair timer;
    start_timer(&timer);
    int block_size_x = 32;
    int block_size_y = 32;
    int numBlocks_x = (N + block_size_x - 1) / block_size_x;
    int numBlocks_y = (M + block_size_y - 1) / (block_size_y);
    dim3 threads(block_size_x, block_size_y);
    dim3 blocks(numBlocks_x, numBlocks_y);

    device_gemm<<<blocks, threads>>>(array_a, array_b, array_c, al, be, M, N, K);


    
    check_launch("gpu graph propagate");
    double gpu_elapsed_time = stop_timer(&timer);

    cudaMemcpy(C, array_c, sizeof(double)*M*N, cudaMemcpyDeviceToHost);
     check_launch("copy from gpu");
    // TODO: free the memory you allocated!
    cudaFree(array_a);
    cudaFree(array_b);
    cudaFree(array_c);
    return 1;
}
