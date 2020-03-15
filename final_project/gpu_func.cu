#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"
#define BLOCK_SIZE 32
#define BLOCK_SIZE_X 4
#define BLOCK_SIZE_Y 16
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
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i < M) && (j < N)) {
        double temp=0;
        for (int k=0; k<K; k++) {
            temp+=A[i+k*M]*B[k+j*K];
        }
        C[i+j*M]=alpha*temp+beta*C[i+j*M];
    }
}


__global__
void device_gemm_shared(double* __restrict__ A, double* __restrict__ B,
           double* __restrict__ C, double alpha, double beta,
           int M, int N, int K) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int rj = threadIdx.x;
    int ri = threadIdx.y;
    double temp=0;
    __shared__ double As[BLOCK_SIZE*BLOCK_SIZE];
    __shared__ double Bs[BLOCK_SIZE*BLOCK_SIZE];

    
    int nb = (K+BLOCK_SIZE-1)/BLOCK_SIZE;
    for (int m=0; m<nb; m++)   {
        if ((i<M) && ((BLOCK_SIZE*m+rj)<K)){
            As[ri+BLOCK_SIZE*rj]=A[i+M*(BLOCK_SIZE*m+rj)];
        }
        if ((j<N) && ((BLOCK_SIZE*m+ri)<K)) {
            Bs[ri+BLOCK_SIZE*rj]=B[BLOCK_SIZE*m+ri+K*j];
        }
        __syncthreads();
        if ((i<M) && (j<N)) {
            for (int k=0; k < BLOCK_SIZE; k++) {
                if ((BLOCK_SIZE*m+k) >= K)  {
                    break;
                }
                temp+=As[ri+BLOCK_SIZE*k]*Bs[k+BLOCK_SIZE*rj];
                
            }
        }
        __syncthreads();
    }
    if ((i<M) && (j<N)) {
            C[i+j*M]=alpha*temp+beta*C[i+j*M];
            printf("C(%d,%d)=%f\n", i, j, C[i+j*M]);
        }
    
}

__global__
void device_gemm_shared2(double* __restrict__ A, double* __restrict__ B,
           double* __restrict__ C, double alpha, double beta,
           int M, int N, int K) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int rj = threadIdx.x;
    int ri = threadIdx.y;
    int row = ri + BLOCK_SIZE_Y * rj;
    int i = blockIdx.y * BLOCK_SIZE_Y * BLOCK_SIZE_X + row;
    __shared__ double Bs[BLOCK_SIZE_X*BLOCK_SIZE_Y];

    double As[BLOCK_SIZE_Y];
    double temp[BLOCK_SIZE_X]={0};

    int nb = (K+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y;
    for (int m=0; m<nb; m++)   {
        if (i<M) {
            for (int ii=0; ii<BLOCK_SIZE_Y;ii++) {
                if ((BLOCK_SIZE_Y*m+ii)>=K) {
                    break;
                }
                As[ii]=A[i+M*(BLOCK_SIZE_Y*m+ii)];
            }
        }
        if ((j<N) && ((BLOCK_SIZE_Y*m+ri)<K)) {
            Bs[ri+BLOCK_SIZE_Y*rj]=B[BLOCK_SIZE_Y*m+ri+K*j];
        }
        __syncthreads();
        if ((i<M)) {
            for (int ii=0; ii<BLOCK_SIZE_X; ii++) {
                if ((blockIdx.x * blockDim.x+ii) >=N) {
                    break;
                }
                for (int k=0; k < BLOCK_SIZE_Y; k++) {
                    if ((BLOCK_SIZE_Y*m+k) >= K)  {
                        break;
                    }
                    temp[ii]+=As[k]*Bs[k+BLOCK_SIZE_Y*ii];
                    
            }
            }
        }
        __syncthreads();
    }
     if ((i<M)) {
            for (int ii=0; ii<BLOCK_SIZE_X; ii++) {
                if ((blockIdx.x * blockDim.x+ii) >=N) {
                    break;
                }
                C[i+M*(blockIdx.x * blockDim.x+ii)]=alpha*temp[ii]+beta*C[i+M*(blockIdx.x * blockDim.x+ii)];
            }
        }
}

__global__
void device_gemm_shared3(double* __restrict__ A, double* __restrict__ B,
           double* __restrict__ C, double alpha, double beta,
           int M, int N, int K) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int rj = threadIdx.x;
    int ri = threadIdx.y;
    int col = ri + BLOCK_SIZE_Y * rj;
    int j = blockIdx.x * BLOCK_SIZE_X * BLOCK_SIZE_Y + col;
    __shared__ double As[BLOCK_SIZE_X*BLOCK_SIZE_Y];

    double Bs[BLOCK_SIZE_X];
    double temp[BLOCK_SIZE_Y]={0};
    int nb = (K+BLOCK_SIZE_X-1)/BLOCK_SIZE_X;
    for (int m=0; m<nb; m++)   {

        if (j<N) {
            for (int ii=0; ii<BLOCK_SIZE_X;ii++) {
                if ((ii+BLOCK_SIZE_X*m)>=K) {
                    break;
                }
                Bs[ii]=B[ii+BLOCK_SIZE_X*m+N*j];
                printf("B(%d, %d)=%f\n", ii+BLOCK_SIZE_X*m, j, B[ii+BLOCK_SIZE_X*m+N*j]);
            }
        }


        if ((i<M) && ((BLOCK_SIZE_X*m+rj)<K)) {
            As[ri+BLOCK_SIZE_Y*rj]=A[i+M*(rj+BLOCK_SIZE_X*m)];
            printf("A(%d, %d)=%f\n", i, rj+BLOCK_SIZE_X*m, A[i+M*(rj+BLOCK_SIZE_X*m)]);
        }


        __syncthreads();

        if ((j<N)) {
            for (int ii=0; ii<BLOCK_SIZE_Y; ii++) {
                if ((blockIdx.y * blockDim.y+ii) >=M) {
                    break;
                }
                for (int k=0; k < BLOCK_SIZE_X; k++) {
                    if ((BLOCK_SIZE_X*m+k) >= K)  {
                        break;
                    }
                    temp[ii]+=As[ii+BLOCK_SIZE_Y*k]*Bs[k];
                    
            }
            }
        }
        __syncthreads();
    }

    if ((j<N)) {
        for (int ii=0; ii<BLOCK_SIZE_Y; ii++) {
            if ((blockIdx.y * blockDim.y+ii) >=M) {
                break;
            }
            C[blockIdx.y * blockDim.y+ii+M*j]=alpha*temp[ii]+beta*C[blockIdx.y * blockDim.y+ii+M*j];
            printf("C(%d,%d)=%f\n", blockIdx.y * blockDim.y+ii, j, C[blockIdx.y * blockDim.y+ii+M*j]);
        }
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
    /*
    int block_size_x = BLOCK_SIZE;
    int block_size_y = BLOCK_SIZE;
    int numBlocks_x = (N + block_size_x - 1) / block_size_x;
    int numBlocks_y = (M + block_size_y - 1) / (block_size_y);
    */
    int block_size_x = BLOCK_SIZE_X;
    int block_size_y = BLOCK_SIZE_Y;
    int numBlocks_x = (N + block_size_x * block_size_y  - 1) / (block_size_y * block_size_x);
    //int numBlocks_x = (N + block_size_x - 1) / (block_size_x);
    int numBlocks_y = (M + block_size_y - 1) / (block_size_y);
    
    dim3 threads(block_size_x, block_size_y);
    dim3 blocks(numBlocks_x, numBlocks_y);
    device_gemm_shared3<<<blocks, threads>>>(A, B, C, al, be, M, N, K);

    block_size_x = BLOCK_SIZE;
    block_size_y = BLOCK_SIZE;
    numBlocks_x = (N + block_size_x - 1) / block_size_x;
    numBlocks_y = (M + block_size_y - 1) / (block_size_y);
    dim3 threads1(block_size_x, block_size_y);
    dim3 blocks1(numBlocks_x, numBlocks_y);
    device_gemm_shared<<<blocks1, threads1>>>(A, B, C, al, be, M, N, K);

    
    return 0;
}
