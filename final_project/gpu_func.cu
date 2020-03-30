#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include <cassert>
#include <math.h>
#include "cublas_v2.h"
#define BLOCK_SIZE 32
#define BLOCK_SIZE_X 5
#define BLOCK_SIZE_Y 17
#define BLOCK_SIZE_K 32
#define BLOCK_SIZE_Z 16
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
           int M, int N, int K, bool A_T=false, bool B_T=false) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i < M) && (j < N)) {
        double temp=0;
        for (int k=0; k<K; k++) {
            double left = A_T ? A[k + i * K] : A[i + k * M];
            double right = B_T ? B[j + k * N] : B[k + j * K];
            temp+=left*right;
/*             if ((i==0) && (j==0)) {
                if (k<=5) printf("A[%d,%d]=%f\n", i, k, A[i + k * M]);
            } */
        }
        C[i + j * M]=alpha*temp+beta*C[i+j*M];
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
        __shared__ double As[BLOCK_SIZE][BLOCK_SIZE+1];
        __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE+1];
        //__shared__ double Accumu[BLOCK_SIZE][BLOCK_SIZE];
        
        int nb = (K+BLOCK_SIZE-1)/BLOCK_SIZE;
        for (int m=0; m<nb; m++)   {
            if (((blockIdx.y * blockDim.y + rj)<M) && ((BLOCK_SIZE*m+ri)<K)){
                //As[ri][rj]=A[i+M*(BLOCK_SIZE*m+rj)];
                As[rj][ri]=A[blockIdx.y * blockDim.y + rj +M*(BLOCK_SIZE*m+ri)];
            }
            if (((blockIdx.x * blockDim.x + ri)<N) && ((BLOCK_SIZE*m+rj)<K)) {
                Bs[rj][ri]=B[BLOCK_SIZE*m+rj+K*(blockIdx.x * blockDim.x + ri)];
            }
            __syncthreads();
            if ((i<M) && (j<N)) {
                for (int k=0; k < BLOCK_SIZE; k++) {
                    if ((BLOCK_SIZE*m+k) >= K)  {
                        break;
                    }
                    temp+=As[ri][k]*Bs[k][rj];
                    //printf("Ctrue(%d,%d, %d)+= %f * %f\n", i, k, j, As[ri+BLOCK_SIZE*k], Bs[k+BLOCK_SIZE*rj]);
                    
                }
            }
            __syncthreads();
        }
        if ((i<M) && (j<N)) {
                C[i+j*M]=alpha*temp+beta*C[i+j*M];
                //printf("Ctrue(%d,%d)=%f\n", i, j, C[i+j*M]);
            }
    
}


__global__
void device_gemm_he(double* __restrict__ A, double* __restrict__ B,
           double* __restrict__ C, double alpha, double beta,
           int M, int N, int K) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int rj = threadIdx.x;
    int ri = threadIdx.y;
    double temp=0;
    __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];
    //__shared__ double Accumu[BLOCK_SIZE][BLOCK_SIZE];
    
    int nb = (K+BLOCK_SIZE-1)/BLOCK_SIZE;
    for (int m=0; m<nb; m++)   {
        if ((i<M) && ((BLOCK_SIZE*m+rj)<K)){
            As[ri][rj]=A[i+M*(BLOCK_SIZE*m+rj)];
        }
        if ((j<N) && ((BLOCK_SIZE*m+ri)<K)) {
            Bs[ri][rj]=B[BLOCK_SIZE*m+ri+K*j];
        }
        __syncthreads();
        if ((i<M) && (j<N)) {
            for (int k=0; k < BLOCK_SIZE; k++) {
                if ((BLOCK_SIZE*m+k) >= K)  {
                    break;
                }
                temp+=As[ri][k]*Bs[k][rj];
                //printf("Ctrue(%d,%d, %d)+= %f * %f\n", i, k, j, As[ri+BLOCK_SIZE*k], Bs[k+BLOCK_SIZE*rj]);
                
            }
        }
        __syncthreads();
    }
    if ((i<M) && (j<N)) {
            C[i+j*M]=alpha*temp+beta*C[i+j*M];
            //printf("Ctrue(%d,%d)=%f\n", i, j, C[i+j*M]);
        }
    
}

__global__
void device_gemm_shared2(double* __restrict__ A, double* __restrict__ B,
           double* __restrict__ C, double alpha, double beta,
           int M, int N, int K) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int rj = threadIdx.x;
    int ri = threadIdx.y;
    int row = ri * BLOCK_SIZE_X + rj;
    int i = blockIdx.y * BLOCK_SIZE_Y * BLOCK_SIZE_X + row;
    __shared__ double Bs[BLOCK_SIZE_Y][BLOCK_SIZE_X+1];

    double As[BLOCK_SIZE_Y];
    double temp[BLOCK_SIZE_X]={0};

    int nb = (K+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y;
    for (int m=0; m<nb; m++)   {
        if ((j<N) && ((BLOCK_SIZE_Y*m+ri)<K)) {
            Bs[ri][rj]=B[BLOCK_SIZE_Y*m+ri+K*j];
        }
        
        __syncthreads();
        if (i<M) {
            for (int ii=0; ii<BLOCK_SIZE_Y;ii++) {
                if ((BLOCK_SIZE_Y*m+ii)>=K) {
                    break;
                }
                As[ii]=A[i+M*(BLOCK_SIZE_Y*m+ii)];
            }
        }
        if ((i<M)) {           
            for (int p = 0; p < BLOCK_SIZE_X * BLOCK_SIZE_Y; p++) {
                int pp = (p + 0 * row) % (BLOCK_SIZE_Y * BLOCK_SIZE_X);
                int ii = pp / BLOCK_SIZE_Y;
                int kk = pp % BLOCK_SIZE_Y;
                if (((blockIdx.x * blockDim.x+ii) >=N) || ((BLOCK_SIZE_Y*m+kk) >= K)) {
                    continue;
                }
                temp[ii]+=As[kk]*Bs[kk][ii];
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
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        int rj = threadIdx.x % BLOCK_SIZE_X;
        int ri = threadIdx.y % BLOCK_SIZE_Y;
        int warp_j = threadIdx.x / BLOCK_SIZE_X;
        int warp_i = threadIdx.y / BLOCK_SIZE_Y;
        int row = ri * BLOCK_SIZE_X + rj;
        int i = blockIdx.y * BLOCK_SIZE_Y * BLOCK_SIZE_X + row;
        int z = warp_j + warp_i * (BLOCK_SIZE / BLOCK_SIZE_X);

        //__shared__ double Bs[BLOCK_SIZE_Y][BLOCK_SIZE_X+1];
        __shared__ double Bs[BLOCK_SIZE_Y][BLOCK_SIZE_X+1];
        __shared__ double Cs[BLOCK_SIZE][BLOCK_SIZE];
        double As[BLOCK_SIZE_Y];
        double temp[BLOCK_SIZE_X]={0};
    
        int nb = (K+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y;
        for (int m=0; m<nb; m++)   {
            if ((j<N) && ((BLOCK_SIZE_Y*m+ri)<K)) {
                Bs[ri][rj]=B[BLOCK_SIZE_Y*m+ri+K*j];
            }
            
            __syncthreads();
            if (i<M) {
                for (int ii=0; ii<BLOCK_SIZE_Y;ii++) {
                    if ((BLOCK_SIZE_Y*m+ii)>=K) {
                        break;
                    }
                    As[ii]=A[i+M*(BLOCK_SIZE_Y*m+ii)];
                }
            }
            if ((i<M)) {           
                for (int p = 0; p < BLOCK_SIZE_X * BLOCK_SIZE_Y; p++) {
                    int pp = (p + 0 * row) % (BLOCK_SIZE_Y * BLOCK_SIZE_X);
                    int ii = pp / BLOCK_SIZE_Y;
                    int kk = pp % BLOCK_SIZE_Y;
                    if (((blockIdx.x * blockDim.x+ii) >=N) || ((BLOCK_SIZE_Y*m+kk) >= K)) {
                        continue;
                    }
                    temp[ii]+=As[kk]*Bs[kk][ii];
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




/*
Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
*/
int myGEMM(double* __restrict__ A, double* __restrict__ B,
           double* __restrict__ C, double* alpha, double* beta,
           int M, int N, int K) {
    /* TODO: Write an efficient GEMM implementation on GPU */
    double al=*alpha;
    double be=*beta;
    
/*     int block_size_x = BLOCK_SIZE;
    int block_size_y = BLOCK_SIZE;
    int numBlocks_x = (N + block_size_x - 1) / block_size_x;
    int numBlocks_y = (M + block_size_y - 1) / (block_size_y); */
    
    int block_size_x = BLOCK_SIZE_X;
    int block_size_y = BLOCK_SIZE_Y;
    int numBlocks_x = (N + block_size_x - 1) / (block_size_x);
    //int numBlocks_x = (N + block_size_y * block_size_x - 1) / (block_size_x * block_size_y);
    int numBlocks_y = (M + block_size_y * block_size_x - 1) / (block_size_x * block_size_y); 
    //printf("myGEMM is called!\n"); 
    dim3 threads(block_size_x, block_size_y);
    dim3 blocks(numBlocks_x, numBlocks_y);
    device_gemm_shared2<<<blocks, threads>>>(A, B, C, al, be, M, N, K);

    
    return 0;
}


int myGEMM2(double* __restrict__ A, double* __restrict__ B,
    double* __restrict__ C, double* alpha, double* beta,
    int M, int N, int K, bool A_T, bool B_T) {
double al=*alpha;
double be=*beta;

int block_size_x = BLOCK_SIZE;
int block_size_y = BLOCK_SIZE;
int numBlocks_x = (N + block_size_x - 1) / block_size_x;
int numBlocks_y = (M + block_size_y - 1) / (block_size_y);
dim3 threads(block_size_x, block_size_y);
dim3 blocks(numBlocks_x, numBlocks_y);
device_gemm<<<blocks, threads>>>(A, B, C, al, be, M, N, K, A_T, B_T);


return 0;
}



/*
Compute C = alpha * A + beta * B
*/

__global__
void device_repmat(double* vec, double* result, int M, int N) {

    
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i < M) && (j < N)) {
        result[i + j * M] = vec[i];
    }
    return;

}

void gpu_repmat(double* vec, double* result, int M, int N)  {

    int block_size_x = BLOCK_SIZE;
    int block_size_y = BLOCK_SIZE;
    int numBlocks_x = (N + block_size_x - 1) / block_size_x;
    int numBlocks_y = (M + block_size_y - 1) / block_size_y;
    dim3 threads(block_size_x, block_size_y);
    dim3 blocks(numBlocks_x, numBlocks_y);
    device_repmat<<<blocks, threads>>>(vec, result, M, N); 
} 


__global__
void device_addmat(double* A, double* B, double* C, double alpha, double beta, int M, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i < M) && (j < N)) {
        C[i + j * M] = alpha * A[i + j * M] + beta * B[i + j * M];
    }
    return;

}


void gpu_addmat(double* dA, double* dB, double* dC, double alpha, double beta, int M, int N)  {

    int block_size_x = BLOCK_SIZE;
    int block_size_y = BLOCK_SIZE;
    int numBlocks_x = (N + block_size_x - 1) / block_size_x;
    int numBlocks_y = (M + block_size_y - 1) / (block_size_y);
    dim3 threads(block_size_x, block_size_y);
    dim3 blocks(numBlocks_x, numBlocks_y);
    device_addmat<<<blocks, threads>>>(dA, dB, dC, alpha, beta, M, N);
}


__global__
void device_sumcol(double* data, double* result, int M, int N) {


    extern __shared__ double sdata[];
    int i = threadIdx.y;
    int j = blockIdx.x;


    if ((i < M) && (j < N)) {
        sdata[i] = data[i + j * M];
    }
    else {
        sdata[i] = 0;
    }
    __syncthreads();
    for (unsigned int s=1; s < blockDim.y; s *= 2) {
        int index = 2 * s * i;
        if ((index + s) < blockDim.y) {
            sdata[index] += sdata[index+s];
        }
        __syncthreads();
    }
        

    if ((i==0) && (j<N)) {
        result[j]=sdata[0];
        //printf("result[%d]=%f\n", j, result[j]);
    }
}

void gpu_sumcol(double* ddata, double* dresult, int M, int N) {

    int block_size_x = 1;
    int block_size_y = 16;
    int numBlocks_x = (N + block_size_x - 1) / block_size_x;
    int numBlocks_y = (M + block_size_y - 1) / (block_size_y);
    dim3 threads(block_size_x, block_size_y);
    dim3 blocks(numBlocks_x, numBlocks_y);
    device_sumcol<<<blocks, threads, block_size_y*sizeof(double)>>>(ddata, dresult, M, N);
}


__global__
void device_sumrow(double* data, double* result, int M, int N) {


    extern __shared__ double sdata[];
    int i = blockIdx.y;
    int j = threadIdx.x;


    if ((i < M) && (j < N)) {
        sdata[j] = data[i + j * M];
    }
    else {
        sdata[j] = 0;
    }
    __syncthreads();
    for (unsigned int s=1; s < blockDim.x; s *= 2) {
        int index = 2 * s * j;
        if ((index) < blockDim.x) {
            sdata[index] += sdata[index+s];
        }
        __syncthreads();
    }
        

    if ((j==0) && (i<M)) {
        result[i]=sdata[0];
        //printf("result[%d]=%f\n", j, result[j]);
    }
}

void gpu_sumrow(double* ddata, double* dresult, int M, int N) {

    int block_size_x = 1024;
    int block_size_y = 1;
    int numBlocks_x = (N + block_size_x - 1) / block_size_x;
    int numBlocks_y = (M + block_size_y - 1) / (block_size_y);
    dim3 threads(block_size_x, block_size_y);
    dim3 blocks(numBlocks_x, numBlocks_y);
    device_sumrow<<<blocks, threads, block_size_x*sizeof(double)>>>(ddata, dresult, M, N);
}


__global__
void device_transpose(double* data, double* result, int M, int N)  {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i < M) && (j < N)) {
        result[j + i * N] = data[i + j * M];
    }
    return; 
}

void gpu_transpose(double* ddata, double* dresult, int M, int N)  {

    int block_size_x = 32;
    int block_size_y = 32;
    int numBlocks_x = (N + block_size_x - 1) / block_size_x;
    int numBlocks_y = (M + block_size_y - 1) / (block_size_y);
    dim3 threads(block_size_x, block_size_y);
    dim3 blocks(numBlocks_x, numBlocks_y);

    device_transpose<<<blocks, threads>>>(ddata, dresult, M, N);
}


__global__
void device_sigmoid(double* data, double* result, int M, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i < M) && (j < N)) {
        result[i + j * M] = 1 / (double)(1+std::exp(-data[i + j * M]));
    }
    return;
}


void gpu_sigmoid(double* ddata, double* dresult, int M, int N)  {

    int block_size_x = BLOCK_SIZE;
    int block_size_y = BLOCK_SIZE;
    int numBlocks_x = (N + block_size_x - 1) / block_size_x;
    int numBlocks_y = (M + block_size_y - 1) / (block_size_y);
    dim3 threads(block_size_x, block_size_y);
    dim3 blocks(numBlocks_x, numBlocks_y);

    device_sigmoid<<<blocks, threads>>>(ddata, dresult, M, N);
}





__global__
void device_exp(double* data, double* result, int M, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i < M) && (j < N)) {
        result[i + j * M] = (double)(std::exp(data[i + j * M]));
/*         if((i==0) &&(j==0)) {
            printf("result[0,0]=%f", result[i+j*M]);
        } */
    }
    return;
}


void gpu_exp(double* ddata, double* dresult, int M, int N) {

    int block_size_x = BLOCK_SIZE;
    int block_size_y = BLOCK_SIZE;
    int numBlocks_x = (N + block_size_x - 1) / block_size_x;
    int numBlocks_y = (M + block_size_y - 1) / (block_size_y);
    dim3 threads(block_size_x, block_size_y);
    dim3 blocks(numBlocks_x, numBlocks_y);

    device_exp<<<blocks, threads>>>(ddata, dresult, M, N);
}


__global__
void device_softmax(double* data, double* result, int M, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i < M) && (j < N)) {
        result[i + j * M] = ((double)(result[i + j * M])/(double)(data[j]));
    }
    return;
}


void gpu_softmax(double* ddata, double* dresult, int M, int N)  {
    int block_size_x = BLOCK_SIZE;
    int block_size_y = BLOCK_SIZE;
    int numBlocks_x = (N + block_size_x - 1) / block_size_x;
    int numBlocks_y = (M + block_size_y - 1) / (block_size_y);
    dim3 threads(block_size_x, block_size_y);
    dim3 blocks(numBlocks_x, numBlocks_y);

    device_softmax<<<blocks, threads>>>(ddata, dresult, M, N);


}


__global__
void device_hadmard(double* c, double* a, double* b, int M, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i < M) && (j < N)) {
/*         if ((j==0)&&(i>495)&&(i<=510)) {
        printf("dz1[%d,%d]=%f\n", i, j, c[i + j * M]);
        } */
        c[i + j * M] = (double)((a[i + j * M]) * (b[i + j * M]) * (1 - b[i + j * M]));
    }
    return;
}


void gpu_hadmard(double* c, double* a, double* b, int M, int N) {

    int block_size_x = BLOCK_SIZE;
    int block_size_y = BLOCK_SIZE;
    int numBlocks_x = (N + block_size_x - 1) / block_size_x;
    int numBlocks_y = (M + block_size_y - 1) / (block_size_y);
    dim3 threads(block_size_x, block_size_y);
    dim3 blocks(numBlocks_x, numBlocks_y);

    device_hadmard<<<blocks, threads>>>(c, a, b, M, N);
}











