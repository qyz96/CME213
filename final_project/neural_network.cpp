#include "neural_network.h"

#include <armadillo>
#include "utils/common.h"
#include "gpu_func.h"
#include "mpi.h"
#include "iomanip"

#define MPI_SAFE_CALL( call ) do {                               \
    int err = call;                                              \
    if (err != MPI_SUCCESS) {                                    \
        fprintf(stderr, "MPI error %d in file '%s' at line %i",  \
               err, __FILE__, __LINE__);                         \
        exit(1);                                                 \
    } } while(0)

double norms(NeuralNetwork& nn) {
    double norm_sum = 0;

    for(int i = 0; i < nn.num_layers; ++i)  {
        norm_sum += arma::accu(arma::square(nn.W[i]));
    }

    return norm_sum;
}

void write_cpudata_tofile(NeuralNetwork& nn, int iter) {
    std::stringstream s;
    s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
    nn.W[0].save(s.str(), arma::raw_ascii);
    std::stringstream t;
    t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
    nn.W[1].save(t.str(), arma::raw_ascii);
    std::stringstream u;
    u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
    nn.b[0].save(u.str(), arma::raw_ascii);
    std::stringstream v;
    v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
    nn.b[1].save(v.str(), arma::raw_ascii);
}

void write_diff_gpu_cpu(NeuralNetwork& nn, int iter,
                        std::ofstream& error_file) {
    arma::mat A, B, C, D;

    std::stringstream s;
    s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
    A.load(s.str(), arma::raw_ascii);
    double max_errW0 = arma::norm(nn.W[0]-A, "inf")/arma::norm(A, "inf");
    double L2_errW0  = arma::norm(nn.W[0]-A,2)/arma::norm(A,2);

    std::stringstream t;
    t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
    B.load(t.str(), arma::raw_ascii);
    double max_errW1 = arma::norm(nn.W[1]-B, "inf")/arma::norm(B, "inf");
    double L2_errW1  = arma::norm(nn.W[1]-B,2)/arma::norm(B,2);

    std::stringstream u;
    u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
    C.load(u.str(), arma::raw_ascii);
    double max_errb0 = arma::norm(nn.b[0]-C, "inf")/arma::norm(C, "inf");
    double L2_errb0  = arma::norm(nn.b[0]-C,2)/arma::norm(C,2);

    std::stringstream v;
    v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
    D.load(v.str(), arma::raw_ascii);
    double max_errb1 = arma::norm(nn.b[1]-D, "inf")/arma::norm(D, "inf");
    double L2_errb1  = arma::norm(nn.b[1]-D,2)/arma::norm(D,2);

    int ow = 15;

    if(iter == 0) {
        error_file << std::left<< std::setw(ow) << "Iteration" << std::left<< std::setw(
                       ow) << "Max Err W0" << std::left << std::setw(ow) << "Max Err W1"
                   << std::left<< std::setw(ow) << "Max Err b0" << std::left<< std::setw(
                       ow) << "Max Err b1" << std::left << std::setw(ow) << "L2 Err W0" << std::left
                   << std::setw(ow) << "L2 Err W1" << std::left<< std::setw(
                       ow) << "L2 Err b0" << std::left<< std::setw(ow) << "L2 Err b1" << "\n";
    }

    error_file << std::left << std::setw(ow) << iter << std::left << std::setw(
                   ow) << max_errW0 << std::left << std::setw(ow) << max_errW1 <<
               std::left << std::setw(ow) << max_errb0 << std::left << std::setw(
                   ow) << max_errb1 << std::left<< std::setw(ow) << L2_errW0 << std::left <<
               std::setw(ow) << L2_errW1 << std::left << std::setw(ow) << L2_errb0 <<
               std::left<< std::setw(ow) << L2_errb1 << "\n";

}

/* CPU IMPLEMENTATIONS */
void feedforward(NeuralNetwork& nn, const arma::mat& X, struct cache& cache) {
    cache.z.resize(2);
    cache.a.resize(2);

    // std::cout << W[0].n_rows << "\n";tw
    assert(X.n_rows == nn.W[0].n_cols);
    cache.X = X;
    int N = X.n_cols;

    arma::mat z1 = nn.W[0] * X + arma::repmat(nn.b[0], 1, N);
    cache.z[0] = z1;

    arma::mat a1;
    sigmoid(z1, a1);
    cache.a[0] = a1;

    assert(a1.n_rows == nn.W[1].n_cols);
    arma::mat z2 = nn.W[1] * a1 + arma::repmat(nn.b[1], 1, N);
    cache.z[1] = z2;

    arma::mat a2;
    softmax(z2, a2);
    cache.a[1] = cache.yc = a2;
}

/*
 * Computes the gradients of the cost w.r.t each param.
 * MUST be called after feedforward since it uses the bpcache.
 * @params y : C x N one-hot column vectors
 * @params bpcache : Output of feedforward.
 * @params bpgrads: Returns the gradients for each param
 */
void backprop(NeuralNetwork& nn, const arma::mat& y, double reg,
              const struct cache& bpcache, struct grads& bpgrads) {
    bpgrads.dW.resize(2);
    bpgrads.db.resize(2);
    int N = y.n_cols;

    //std::cout << "a0 size " << bpcache.a[0].n_elem << "\n";
    arma::mat diff = (1.0 / N) * (bpcache.yc - y);
    bpgrads.dW[1] = diff * bpcache.a[0].t() + reg * nn.W[1];
    bpgrads.db[1] = arma::sum(diff, 1);
    //std::cout << "diff size " << diff.n_elem << "\n";
    arma::mat da1 = nn.W[1].t() * diff;

    arma::mat dz1 = da1 % bpcache.a[0] % (1 - bpcache.a[0]);
    //std::cout << "X size " << bpcache.X.n_elem << "\n";
    bpgrads.dW[0] = dz1 * bpcache.X.t() + reg * nn.W[0];
    bpgrads.db[0] = arma::sum(dz1, 1);
}

/*
 * Computes the Cross-Entropy loss function for the neural network.
 */
double loss(NeuralNetwork& nn, const arma::mat& yc, const arma::mat& y,
            double reg) {
    int N = yc.n_cols;
    double ce_sum = -arma::accu(arma::log(yc.elem(arma::find(y == 1))));

    double data_loss = ce_sum / N;
    double reg_loss = 0.5 * reg * norms(nn);
    double loss = data_loss + reg_loss;
    // std::cout << "Loss: " << loss << "\n";
    return loss;
}

/*
 * Returns a vector of labels for each row vector in the input
 */
void predict(NeuralNetwork& nn, const arma::mat& X, arma::rowvec& label) {
    struct cache fcache;
    feedforward(nn, X, fcache);
    label.set_size(X.n_cols);

    for(int i = 0; i < X.n_cols; ++i) {
        arma::uword row;
        fcache.yc.col(i).max(row);
        label(i) = row;
    }
}

/*
 * Computes the numerical gradient
 */
void numgrad(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
             double reg, struct grads& numgrads) {
    double h = 0.00001;
    struct cache numcache;
    numgrads.dW.resize(nn.num_layers);
    numgrads.db.resize(nn.num_layers);

    for(int i = 0; i < nn.num_layers; ++i) {
        numgrads.dW[i].resize(nn.W[i].n_rows, nn.W[i].n_cols);

        for(int j = 0; j < nn.W[i].n_rows; ++j) {
            for(int k = 0; k < nn.W[i].n_cols; ++k) {
                double oldval = nn.W[i](j,k);
                nn.W[i](j, k) = oldval + h;
                feedforward(nn, X, numcache);
                double fxph = loss(nn, numcache.yc, y, reg);
                nn.W[i](j, k) = oldval - h;
                feedforward(nn, X, numcache);
                double fxnh = loss(nn, numcache.yc, y, reg);
                numgrads.dW[i](j, k) = (fxph - fxnh) / (2*h);
                nn.W[i](j, k) = oldval;
            }
        }
    }

    for(int i = 0; i < nn.num_layers; ++i) {
        numgrads.db[i].resize(nn.b[i].n_rows, nn.b[i].n_cols);

        for(int j = 0; j < nn.b[i].size(); ++j) {
            double oldval = nn.b[i](j);
            nn.b[i](j) = oldval + h;
            feedforward(nn, X, numcache);
            double fxph = loss(nn, numcache.yc, y, reg);
            nn.b[i](j) = oldval - h;
            feedforward(nn, X, numcache);
            double fxnh = loss(nn, numcache.yc, y, reg);
            numgrads.db[i](j) = (fxph - fxnh) / (2*h);
            nn.b[i](j) = oldval;
        }
    }
}

/*
 * Train the neural network &nn
 */
void train(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
           double learning_rate, double reg,
           const int epochs, const int batch_size, bool grad_check, int print_every,
           int debug) {
    int N = X.n_cols;
    int iter = 0;
    int print_flag = 0;

    for(int epoch = 0 ; epoch < epochs; ++epoch) {
        int num_batches = (N + batch_size - 1)/batch_size;

        for(int batch = 0; batch < num_batches; ++batch) {
            int last_col = std::min((batch + 1)*batch_size-1, N-1);
            arma::mat X_batch = X.cols(batch * batch_size, last_col);
            arma::mat y_batch = y.cols(batch * batch_size, last_col);

            struct cache bpcache;
            feedforward(nn, X_batch, bpcache);

            struct grads bpgrads;
            backprop(nn, y_batch, reg, bpcache, bpgrads);

            if(print_every > 0 && iter % print_every == 0) {
                if(grad_check) {
                    struct grads numgrads;
                    numgrad(nn, X_batch, y_batch, reg, numgrads);
                    assert(gradcheck(numgrads, bpgrads));
                }

                std::cout << "Loss at iteration " << iter << " of epoch " << epoch << "/" <<
                          epochs << " = " << loss(nn, bpcache.yc, y_batch, reg) << "\n";
            }

            // Gradient descent step
            for(int i = 0; i < nn.W.size(); ++i) {
                nn.W[i] -= learning_rate * bpgrads.dW[i];
            }

            for(int i = 0; i < nn.b.size(); ++i) {
                nn.b[i] -= learning_rate * bpgrads.db[i];
            }

            /* Debug routine runs only when debug flag is set. If print_every is zero, it saves
               for the first batch of each epoch to avoid saving too many large files.
               Note that for the first time, you have to run debug and serial modes together.
               This will run the following function and write out files to CPUmats folder.
               In the later runs (with same parameters), you can use just the debug flag to
               output diff b/w CPU and GPU without running CPU version */
            if(print_every <= 0) {
                print_flag = batch == 0;
            } else {
                print_flag = iter % print_every == 0;
            }

            if(debug && print_flag) {
                write_cpudata_tofile(nn, iter);
            }

            iter++;
        }
    }
}






void gpu_feedforward(NeuralNetwork& nn, const arma::mat& X, struct cache& bpcache) {


    int K = nn.W[0].n_rows;
    int num_sample = X.n_cols;
    int M = nn.W[0].n_cols;
    int N = nn.W[1].n_rows;
    bpcache.z.resize(2);
    bpcache.a.resize(2);
    arma::mat b0r = arma::repmat(nn.b[0], 1, N);
    arma::mat b1r = arma::repmat(nn.b[1], 1, N);

    double* a0;
    double* a1;
    double* z0;
    double* z1;
    double* yc;
    a0 = (double*)malloc(K*num_sample*sizeof(double));
    a1 = (double*)malloc(N*num_sample*sizeof(double));
    z0 = (double*)malloc(K*num_sample*sizeof(double));
    z1 = (double*)malloc(N*num_sample*sizeof(double));
    yc = (double*)malloc(N*num_sample*sizeof(double));
    double* W1_test=nn.W[1].memptr();
    double* W0_test=nn.W[0].memptr();
    double* W3_test=(double*)malloc(N*K*sizeof(double));






    //my_feedforward(nn, X, bpcache, b0r, b1r, a0, a1, z0, z1, yc);
    double* dz0;
    double* dz1;
    double* da0;
    double* da1;
    double* dW0;
    double* dW1;
    double* db0;
    double* db1;
    double* dX;
    double* dexp;

    


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
    cudaMalloc((void**)&dexp, sizeof(double) * 1 * num_sample);

    
    //std::cout<<"Copying CUDA memory....\n";
    cudaMemcpy(dz0, b0r.memptr(), sizeof(double) * K * num_sample , cudaMemcpyHostToDevice);
    cudaMemcpy(dz1, b1r.memptr(), sizeof(double) * N * num_sample, cudaMemcpyHostToDevice);
    cudaMemcpy(da0, a0, sizeof(double) * K * num_sample, cudaMemcpyHostToDevice);
    cudaMemcpy(da1, a1, sizeof(double) * N * num_sample, cudaMemcpyHostToDevice);


    //cudaMemcpy(dW0, nn.W[0].memptr(), sizeof(double) * M * K, cudaMemcpyHostToDevice);


    cudaMemcpy(dW0, nn.W[0].memptr(), sizeof(double) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(dW1, nn.W[1].memptr(), sizeof(double) * K * N, cudaMemcpyHostToDevice);
    cudaMemcpy(dX, X.memptr(), sizeof(double) * M * num_sample, cudaMemcpyHostToDevice);



    //std::cout<<"nn.W[0] * X + arma::repmat(nn.b[0], 1, N)....\n";
    double alpha = 1;
    double beta = 1;


    myGEMM(dW0, dX, dz0, &alpha, &beta, K, num_sample, M);
    cudaMemcpy(z0, dz0, sizeof(double) * K * num_sample, cudaMemcpyDeviceToHost);
    check_launch("myGEMM 1");
    gpu_sigmoid(dz0, da0, K, num_sample);
    cudaMemcpy(a0, da0, sizeof(double) * K * num_sample, cudaMemcpyDeviceToHost);
    std::cout<<"nn.W[0] * X + arma::repmat(nn.b[0], 1, N)....\n";
    myGEMM(dW1, da0, dz1, &alpha, &beta, N, num_sample, K);
    check_launch("myGEMM 2");
    cudaMemcpy(z1, dz1, sizeof(double) * N * num_sample, cudaMemcpyDeviceToHost);
    
    gpu_exp(dz1, da1, N, num_sample);
    check_launch("exp");
    cudaMemcpy(a1, da1, sizeof(double) * N * num_sample, cudaMemcpyDeviceToHost);
    gpu_sumcol(da1, dexp, N, num_sample);
    check_launch("sumcol");
    gpu_softmax(dexp, da1, N, num_sample);
    cudaMemcpy(a1, da1, sizeof(double) * N * num_sample, cudaMemcpyDeviceToHost);
    cudaMemcpy(yc, da1, sizeof(double) * N * num_sample, cudaMemcpyDeviceToHost);
    
    cudaFree(dz0);
    cudaFree(dz1);
    cudaFree(da0);
    cudaFree(da1);
    cudaFree(dW0);
    cudaFree(dW1);
    cudaFree(db0);
    cudaFree(db1);
    cudaFree(dX);
    cudaFree(dexp);







    bpcache.z[0]=arma::mat(z0, K, num_sample);
    bpcache.a[0]=arma::mat(a0, K, num_sample);
    //std::cout<<"z0: "<<bpcache.z[0].submat(0,0,5,5)<<"\n";
    bpcache.z[1]=arma::mat(z1, N, num_sample);
    //std::cout<<bpcache.z[1]<<"\n";
    bpcache.a[1]=arma::mat(a1, N, num_sample);
    //std::cout<<bpcache.a[1]<<"\n";
    bpcache.yc = arma::mat(yc, N, num_sample);
    arma::mat W1t = arma::mat(W3_test, N, K);
    //std::cout<<"W1t: "<<W1t.submat(0,0, 5, 5)<<"\n";
    bpcache.X = X;
    free(a0);
    free(a1);
    free(z0);
    free(z1);
    free(yc);

}



void gpu_backprop(NeuralNetwork& nn, const arma::mat& y, double reg, const struct cache& bpcache, struct grads& bpgrads) {
    int num_sample = y.n_cols;
    int K = nn.W[0].n_rows;
    int M = nn.W[0].n_cols;
    int N = nn.W[1].n_rows;
    bpgrads.dW.resize(2);
    bpgrads.db.resize(2);


    double* W0;
    double* W1;
    double* b0;
    double* b1;

    W0 = (double*)(malloc(M*K*sizeof(double)));
    W1 = (double*)(malloc(N*K*sizeof(double)));
    b0 = (double*)(malloc(K*sizeof(double)));
    b1 = (double*)(malloc(N*sizeof(double)));

    double* dW0;
    double* da0;
    double* dW1;
    double* db0;
    double* db1;
    double* dyc;
    double* dy;
    double* dDff;
    double* da1;
    double* dz1;
    double* dX;

    cudaMalloc((void**)&dW0, sizeof(double) * M * K);
    cudaMalloc((void**)&dW1, sizeof(double) * K * N);
    cudaMalloc((void**)&db0, sizeof(double) * K);
    cudaMalloc((void**)&db1, sizeof(double) * N);
    cudaMalloc((void**)&dyc, sizeof(double) * N * num_sample);
    cudaMalloc((void**)&dy, sizeof(double) * N * num_sample);
    cudaMalloc((void**)&dDff, sizeof(double) * N * num_sample);
    cudaMalloc((void**)&da1, sizeof(double) * K * num_sample);
    cudaMalloc((void**)&dz1, sizeof(double) * K * num_sample);
    cudaMalloc((void**)&dX, sizeof(double) * M * num_sample);
    cudaMalloc((void**)&da0, sizeof(double) * K * num_sample);

    cudaMemcpy(dW0, nn.W[0].memptr(), sizeof(double) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(dW1, nn.W[1].memptr(), sizeof(double) * K * N, cudaMemcpyHostToDevice);
    cudaMemcpy(dyc, bpcache.yc.memptr(), sizeof(double) * N * num_sample, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y.memptr(), sizeof(double) * N * num_sample, cudaMemcpyHostToDevice);
    cudaMemcpy(dX, bpcache.X.memptr(), sizeof(double) * M * num_sample, cudaMemcpyHostToDevice);
    cudaMemcpy(da0, bpcache.a[0].memptr(), sizeof(double) * K * num_sample, cudaMemcpyHostToDevice);


    


    double alpha = 1/(double)(num_sample);
    double beta = -1/(double)(num_sample);
    double alpha1 = 1;
    double beta1=0;
    
    gpu_addmat(dyc, dy, dDff, 1/(double)(num_sample), -1/(double)(num_sample), N, num_sample);

    myGEMM2(dW1, dDff, da1, &alpha1, &beta1, K, N, num_sample, true, false);
    myGEMM2(dDff, da0, dW1, &alpha1, &reg, N, K, num_sample, false, true);

    gpu_hadmard(dz1, da1, da0, K, num_sample);

    myGEMM2(dy, da0, dW1, &alpha1, &reg, N, num_sample, K, false, false);



    gpu_sumrow(dDff, db1, N, num_sample);
    myGEMM2(dz1, dX, dW0, &alpha1, &reg, K, M, num_sample, false, true);
    gpu_sumrow(dz1, db0, K, num_sample);

    
    


    cudaMemcpy(W0, dW0, sizeof(double) * M * K, cudaMemcpyDeviceToHost);
    cudaMemcpy(b0, db0, sizeof(double) * K, cudaMemcpyDeviceToHost);
    cudaMemcpy(W1, dW1, sizeof(double) * N * K, cudaMemcpyDeviceToHost);
    cudaMemcpy(b1, db1, sizeof(double) * N, cudaMemcpyDeviceToHost);


    cudaFree(dW0);
    cudaFree(da0);
    cudaFree(dW1);
    cudaFree(db0);
    cudaFree(db1);
    cudaFree(dyc);
    cudaFree(dy);
    cudaFree(dDff);
    cudaFree(da1);
    cudaFree(dz1);
    cudaFree(dX);

    bpgrads.dW[0]=arma::mat(W0, K, M);
    bpgrads.dW[1]=arma::mat(W1, N, K);
    bpgrads.db[0]=arma::vec(b0, K);
    bpgrads.db[1]=arma::vec(b1, N);
    
/*     arma::mat diff = (1.0 / N) * (bpcache.yc - y);
    bpgrads.dW[1] = diff * bpcache.a[0].t() + reg * nn.W[1];
    bpgrads.db[1] = arma::sum(diff, 1);
    arma::mat da1 = nn.W[1].t() * diff;

    arma::mat dz1 = da1 % bpcache.a[0] % (1 - bpcache.a[0]);

    bpgrads.dW[0] = dz1 * bpcache.X.t() + reg * nn.W[0];
    bpgrads.db[0] = arma::sum(dz1, 1); */

    free(W0);
    free(W1);
    free(b0);
    free(b1);
    


}





/*
 * TODO
 * Train the neural network &nn of rank 0 in parallel. Your MPI implementation
 * should mainly be in this function.
 */
void parallel_train(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
                    double learning_rate, double reg,
                    const int epochs, const int batch_size, bool grad_check, int print_every,
                    int debug) {

    int rank, num_procs;
    MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
    MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    int N = (rank == 0)?X.n_cols:0;
    MPI_SAFE_CALL(MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD));

    std::ofstream error_file;
    error_file.open("Outputs/CpuGpuDiff.txt");
    int print_flag = 0;

    /* HINT: You can obtain a raw pointer to the memory used by Armadillo Matrices
       for storing elements in a column major way. Or you can allocate your own array
       memory space and store the elements in a row major way. Remember to update the
       Armadillo matrices in NeuralNetwork &nn of rank 0 before returning from the function. */

    /* iter is a variable used to manage debugging. It increments in the inner loop
       and therefore goes from 0 to epochs*num_batches */
    int iter = 0;
    print_every=1;
    grad_check=true;
    for(int epoch = 0; epoch < epochs; ++epoch) {
        int num_batches = (N + batch_size - 1)/batch_size;

        for(int batch = 0; batch < num_batches; ++batch) {
            /*
             * Possible Implementation:
             * 1. subdivide input batch of images and `MPI_scatter()' to each MPI node
             * 2. compute each sub-batch of images' contribution to network coefficient updates
             * 3. reduce the coefficient updates and broadcast to all nodes with `MPI_Allreduce()'
             * 4. update local network coefficient at each node
             */

            int last_col = std::min((batch + 1)*batch_size-1, N-1);
            arma::mat X_batch = X.cols(batch * batch_size, last_col);
            arma::mat y_batch = y.cols(batch * batch_size, last_col);

            struct cache bpcache;


            gpu_feedforward(nn, X_batch, bpcache);
            std::cout<<"nnW0: "<<nn.W[0].submat(0,0, 5, 5)<<"\n";
            std::cout<<"nnW1: "<<nn.W[1].submat(0,0, 5, 5)<<"\n";
            std::cout<<"z0: "<<bpcache.z[0].submat(0,0, 5, 5)<<"\n";
            std::cout<<"a0: "<<bpcache.a[0].submat(0,0,5,5)<<"\n";
            std::cout<<"z1: "<<bpcache.z[1].submat(0,0,5,5)<<"\n";
            std::cout<<"a1: "<<bpcache.a[1].submat(0,0,5,5)<<"\n";
            std::cout<<"y: "<<bpcache.yc.submat(0,0, 5, 5)<<"\n";
            struct grads bpgrads;
            //std::cout<<"Backpropagation begins...\n";
            gpu_backprop(nn, y_batch, reg, bpcache, bpgrads);

            //backprop(nn, y_batch, reg, bpcache, bpgrads);
            std::cout<<"dW0: "<<bpgrads.dW[0].submat(0,0, 5, 5)<<"\n";
            std::cout<<"dW1: "<<bpgrads.dW[1].submat(0,0, 5, 5)<<"\n";
            std::cout<<"b0: "<<bpgrads.db[0].subvec(0, 5)<<"\n";
            std::cout<<"b1: "<<bpgrads.db[1].subvec(0,5)<<"\n";
            //std::cout<<"Backpropagation done...\n";
            if(print_every > 0 && iter % print_every == 0) {
                if(grad_check) {
                    struct grads numgrads;
                    numgrad(nn, X_batch, y_batch, reg, numgrads);
                    assert(gradcheck(numgrads, bpgrads));
                }

                std::cout << "Loss at iteration " << iter << " of epoch " << epoch << "/" <<
                          epochs << " = " << loss(nn, bpcache.yc, y_batch, reg) << "\n";
                return;
            }
            //std::cout<<"Subtracting gradient...\n";
            // Gradient descent step
            for(int i = 0; i < nn.W.size(); ++i) {
                nn.W[i] -= learning_rate * bpgrads.dW[i];
            }

            for(int i = 0; i < nn.b.size(); ++i) {
                nn.b[i] -= learning_rate * bpgrads.db[i];
            }
            //std::cout<<"Subtracting gradient done...\n";
            if(print_every <= 0) {
                print_flag = batch == 0;
            } else {
                print_flag = iter % print_every == 0;
            }

            /* Following debug routine assumes that you have already updated the arma
               matrices in the NeuralNetwork nn.  */
            if(debug && rank == 0 && print_flag) {
                write_diff_gpu_cpu(nn, iter, error_file);
            }

            iter++;
        }
    }

    error_file.close();
}
