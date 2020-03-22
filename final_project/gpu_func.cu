#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include <cassert>
#include <math.h>
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
    printf("device_gemm is called!\n");
    if ((i < M) && (j < N)) {
        double temp=0;
        for (int k=0; k<K; k++) {
            temp+=A[i+k*M]*B[k+j*K];
            if ((i==0) && (j==0)) {

                if (k<=5) {printf("w[%d,%d]=%f\n", i, k, A[i+k*M]);}
            }
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
    //int col = ri + BLOCK_SIZE_Y * rj;
    int col = ri * BLOCK_SIZE_X + rj;
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
                Bs[ii]=B[ii+BLOCK_SIZE_X*m+K*j];
                //printf("Bs[%d]=B(%d, %d)=%f\n", ii, ii+BLOCK_SIZE_X*m, j, B[ii+BLOCK_SIZE_X*m+N*j]);
            }
        }


        if ((i<M) && ((BLOCK_SIZE_X*m+rj)<K)) {
            As[ri+BLOCK_SIZE_Y*rj]=A[i+M*(rj+BLOCK_SIZE_X*m)];
            //printf("A(%d, %d)=%f\n", i, rj+BLOCK_SIZE_X*m, A[i+M*(rj+BLOCK_SIZE_X*m)]);
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
                    //printf("C(%d, %d, %d)+= %f * %f\n", blockIdx.y * blockDim.y+ii, BLOCK_SIZE_X*m+k, j, As[ii+BLOCK_SIZE_Y*k],Bs[k]);
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
            //printf("C(%d,%d)=%f\n", blockIdx.y * blockDim.y+ii, j, C[blockIdx.y * blockDim.y+ii+M*j]);
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
    
    int block_size_x = BLOCK_SIZE;
    int block_size_y = BLOCK_SIZE;
    int numBlocks_x = (N + block_size_x - 1) / block_size_x;
    int numBlocks_y = (M + block_size_y - 1) / (block_size_y);
    
/*     int block_size_x = BLOCK_SIZE_X;
    int block_size_y = BLOCK_SIZE_Y;
    int numBlocks_x = (N + block_size_x * block_size_y  - 1) / (block_size_y * block_size_x);
    int numBlocks_x = (N + block_size_x - 1) / (block_size_x);
    int numBlocks_y = (M + block_size_y - 1) / (block_size_y); */
    printf("myGEMM is called!\n");
    dim3 threads(block_size_x, block_size_y);
    dim3 blocks(numBlocks_x, numBlocks_y);
    device_gemm<<<blocks, threads>>>(A, B, C, al, be, M, N, K);
    /*
    block_size_x = BLOCK_SIZE;
    block_size_y = BLOCK_SIZE;
    numBlocks_x = (N + block_size_x - 1) / block_size_x;
    numBlocks_y = (M + block_size_y - 1) / (block_size_y);
    dim3 threads1(block_size_x, block_size_y);
    dim3 blocks1(numBlocks_x, numBlocks_y);
    device_gemm_shared<<<blocks1, threads1>>>(A, B, C, al, be, M, N, K);
    */
    
    return 0;
}


/*
Compute C = alpha * A + beta * B
*/

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
    if (j < N) {
        if (i < M) {
            sdata[i] = data[i + j * M];
        }
        else {
            sdata[i] = 0;
        }
        __syncthreads();
        for (unsigned int s=0; s < blockDim.y; s *= 2) {
            int index = 2 * s * i;
            if (index < blockDim.y) {
                sdata[index] += sdata[index+s];
            }
        }
        __syncthreads();


        result[j]=sdata[0];
    }
}

void gpu_sumcol(double* ddata, double* dresult, int M, int N) {

    int block_size_x = 1;
    int block_size_y = 16;
    int numBlocks_x = (N + block_size_x - 1) / block_size_x;
    int numBlocks_y = (M + block_size_y - 1) / (block_size_y);
    dim3 threads(block_size_x, block_size_y);
    dim3 blocks(numBlocks_x, numBlocks_y);
    device_sumcol<<<blocks, threads>>>(ddata, dresult, M, N);
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

    device_sigmoid<<<blocks, threads>>>(ddata, dresult, M, N);
}


__global__
void device_softmax(double* data, double* result, int M, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i < M) && (j < N)) {
        result[i + j * M] = (double)((result[i + j * M])/(data[j]));
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




void my_feedforward(NeuralNetwork& nn, const arma::mat& X, struct cache& cache, 
    const arma::mat& b0r, const arma::mat& b1r, const arma::mat& T, double* a0, double* a1, double* z0, double* z1, double* yc, double* W1_test, double* W0_test, double* W3_test) {

    double* dz0;
    double* dz1;
    double* da0;
    double* da1;
    double* dW0;
    double* dW1;
    double* db0;
    double* db1;
    double* dX;
    double* dT;
    double* dexp;

    
    int num_sample = X.n_cols;
    int K = nn.W[0].n_rows;
    int M = nn.W[0].n_cols;
    int N = nn.W[1].n_rows;


    //std::cout<<"Allocating CUDA memory....\n";
    cudaMalloc((void**)&dz0, sizeof(double) * K * num_sample);
    cudaMalloc((void**)&dz1, sizeof(double) * N * num_sample);
    cudaMalloc((void**)&da0, sizeof(double) * K * num_sample);
    cudaMalloc((void**)&da1, sizeof(double) * N * num_sample);
    cudaMalloc((void**)&dW0, sizeof(double) * M * K);
    cudaMalloc((void**)&dW1, sizeof(double) * K * N);
    cudaMalloc((void**)&db0, sizeof(double) * K * num_sample);
    cudaMalloc((void**)&db1, sizeof(double) * N * num_sample);
    cudaMalloc((void**)&dX, sizeof(double) * K * num_sample);
    cudaMalloc((void**)&dT, sizeof(double) * N * num_sample);
    cudaMalloc((void**)&dexp, sizeof(double) * 1 * num_sample);

    
    //std::cout<<"Copying CUDA memory....\n";
    cudaMemcpy(dz0, b0r.memptr(), sizeof(double) * K * num_sample , cudaMemcpyHostToDevice);
    cudaMemcpy(dz1, b1r.memptr(), sizeof(double) * N * num_sample, cudaMemcpyHostToDevice);
    cudaMemcpy(da0, a0, sizeof(double) * K * num_sample, cudaMemcpyHostToDevice);
    cudaMemcpy(da1, a1, sizeof(double) * N * num_sample, cudaMemcpyHostToDevice);


    //cudaMemcpy(dW0, nn.W[0].memptr(), sizeof(double) * M * K, cudaMemcpyHostToDevice);


    cudaMemcpy(dW0, nn.W[0].memptr(), sizeof(double) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(dW1, nn.W[1].memptr(), sizeof(double) * K * N, cudaMemcpyHostToDevice);
    cudaMemcpy(dT, T.memptr(), sizeof(double) * N * num_sample, cudaMemcpyHostToDevice);
    cudaMemcpy(dX, X.memptr(), sizeof(double) * M * num_sample, cudaMemcpyHostToDevice);



    //std::cout<<"nn.W[0] * X + arma::repmat(nn.b[0], 1, N)....\n";
    double alpha = 1;
    double beta = 1;


    myGEMM(dW0, dX, dz0, &alpha, &beta, K, num_sample, M);
    gpu_sigmoid(dz0, da0, K, num_sample);
    std::cout<<"nn.W[0] * X + arma::repmat(nn.b[0], 1, N)....\n";
    myGEMM(dW1, da0, dz1, &alpha, &beta, N, num_sample, K);

    
    gpu_exp(dz1, da1, N, num_sample);
    gpu_sumcol(da1, dexp, N, num_sample);
    gpu_softmax(dexp, da1, N, num_sample);
    cudaMemcpy(a1, da1, sizeof(double) * N * num_sample, cudaMemcpyDeviceToHost);
    cudaMemcpy(yc, da1, sizeof(double) * N * num_sample, cudaMemcpyDeviceToHost);
    cudaMemcpy(z0, dz0, sizeof(double) * K * num_sample, cudaMemcpyDeviceToHost);
    cudaMemcpy(a0, da0, sizeof(double) * K * num_sample, cudaMemcpyDeviceToHost);
    cudaMemcpy(W3_test, dW1, sizeof(double) * N * K, cudaMemcpyDeviceToHost);
    cudaMemcpy(z1, dz1, sizeof(double) * N * num_sample, cudaMemcpyDeviceToHost);



}





void my_backprop(NeuralNetwork& nn, const arma::mat& y, double reg, const struct cache& bpcache, struct grads& bpgrads) {
    int num_sample = bpcache.X.n_cols;
    int K = nn.W[0].n_rows;
    int M = nn.W[0].n_cols;
    int N = nn.W[1].n_rows;
    
    bpgrads.dW.resize(2);
    bpgrads.dW[0].zeros(K, M);
    bpgrads.dW[1].zeros(N, K);
    bpgrads.db.resize(2);
    bpgrads.db[0].zeros(K);
    bpgrads.db[1].zeros(N);

    arma::vec allones = arma::ones<arma::vec>(num_sample);

    double* dW0;
    double* da0;
    double* dW1;
    double* db0;
    double* db1;
    double* dyc;
    double* dy;
    double* dDff;
    double* dOne;
    double* daz;
    double* dX;

    cudaMalloc((void**)&dW0, sizeof(double) * M * K);
    cudaMalloc((void**)&dW1, sizeof(double) * K * N);
    cudaMalloc((void**)&db0, sizeof(double) * K);
    cudaMalloc((void**)&db1, sizeof(double) * N);
    cudaMalloc((void**)&dyc, sizeof(double) * N * num_sample);
    cudaMalloc((void**)&dy, sizeof(double) * N * num_sample);
    cudaMalloc((void**)&dOne, sizeof(double) * num_sample);
    cudaMalloc((void**)&daz, sizeof(double) * K * num_sample);
    cudaMalloc((void**)&dX, sizeof(double) * M * num_sample);
    cudaMalloc((void**)&da0, sizeof(double) * K * num_sample);

    cudaMemcpy(dW0, nn.W[0].memptr(), sizeof(double) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(dW1, nn.W[1].memptr(), sizeof(double) * K * N, cudaMemcpyHostToDevice);
    cudaMemcpy(db0, nn.b[0].memptr(), sizeof(double) * K, cudaMemcpyHostToDevice);
    cudaMemcpy(db1, nn.b[1].memptr(), sizeof(double) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(dyc, bpcache.yc.memptr(), sizeof(double) * N * num_sample, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y.memptr(), sizeof(double) * N * num_sample, cudaMemcpyHostToDevice);
    cudaMemcpy(dOne, allones.memptr(), sizeof(double) * num_sample, cudaMemcpyHostToDevice);
    cudaMemcpy(dX, bpcache.X.memptr(), sizeof(double) * M * num_sample, cudaMemcpyHostToDevice);
    cudaMemcpy(da0, bpcache.a[0].memptr(), sizeof(double) * K * num_sample, cudaMemcpyHostToDevice);


    int block_size_x = 32;
    int block_size_y = 32;
    int numBlocks_x = (num_sample + block_size_x - 1) / block_size_x;
    int numBlocks_y = (K + block_size_y - 1) / (block_size_y);
    dim3 threads(block_size_x, block_size_y);
    dim3 blocks(numBlocks_x, numBlocks_y);
    double alpha = 1/(double)(num_sample);
    double beta = -1/(double)(num_sample);
    double alpha1 = 1;
    double beta1=0;
    device_addmat<<<blocks, threads>>>(dyc, dy, dy, alpha, beta, N, num_sample);

    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&handle);
    /*
    stat = cublasDgemm(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        K, num_sample, N,
        &alpha1,
        dW1, K,
        dDff, N,
        &beta1,
        daz, K);

    */
    myGEMM(dW1, dDff, daz, &alpha1, &beta1, K, N, num_sample);


    /*
    stat = cublasDgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        N, K, num_sample,
        &alpha1,
        dy, N,
        da0, num_sample,
        &reg,
        dW1, N);

    */

    myGEMM(dy, da0, dW1, &alpha1, &reg, N, num_sample, K);


    /*

    stat = cublasDgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, 1, num_sample,
        &alpha1,
        dDff, N,
        dOne, num_sample,
        &beta1,
        db1, N);
    */
    
    myGEMM(dDff, dOne, db1, &alpha1, &beta1, N, 1, num_sample);
    device_hadmard<<<blocks, threads>>>(daz, daz, da0, K, num_sample);

    stat = cublasDgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        K, M, num_sample,
        &alpha1,
        daz, K,
        dX, num_sample,
        &reg,
        dW0, N);



    myGEMM(daz, dX, dW0, &alpha1, &beta1, N, 1, num_sample);
    stat = cublasDgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        K, num_sample, 1,
        &alpha1,
        daz, K,
        dOne, num_sample,
        &beta1,
        db0, K);

    myGEMM(daz, dOne, db0, &alpha1, &beta1, K, num_sample, 1);
    
    //std::cout << "backprop " << bpcache.yc << "\n";


    cudaMemcpy(bpgrads.dW[0].memptr(), dW0, sizeof(double) * M * K, cudaMemcpyDeviceToHost);
    cudaMemcpy(bpgrads.db[0].memptr(), db0, sizeof(double) * K, cudaMemcpyDeviceToHost);
    cudaMemcpy(bpgrads.dW[1].memptr(), dW1, sizeof(double) * N * K, cudaMemcpyDeviceToHost);
    cudaMemcpy(bpgrads.db[1].memptr(), db1, sizeof(double) * N, cudaMemcpyDeviceToHost);

    /*
    arma::mat diff = (1.0 / N) * (bpcache.yc - y);
    bpgrads.dW[1] = diff * bpcache.a[0].t() + reg * nn.W[1];
    bpgrads.db[1] = arma::sum(diff, 1);
    arma::mat da1 = nn.W[1].t() * diff;

    arma::mat dz1 = da1 % bpcache.a[0] % (1 - bpcache.a[0]);

    bpgrads.dW[0] = dz1 * bpcache.X.t() + reg * nn.W[0];
    bpgrads.db[0] = arma::sum(dz1, 1);

    */


}
