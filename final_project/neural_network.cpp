#include "neural_network.h"

#include <armadillo>
#include "utils/common.h"
#include "cublas_v2.h"
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
/*             std::cout<<"x_subbatch: "<<X_batch.submat(0,0,5,5)<<"\n";
            std::cout<<"y_subbatch: "<<y_batch.submat(0,0,5,5)<<"\n"; */
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
/*             std::cout<<"serial b0: "<<nn.b[0].subvec(0,  5)<<"\n";
            if (batch > 5) {
                return;
            } */

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




class OneBatchUpdate2  {



    public:
    OneBatchUpdate2(NeuralNetwork& nn, int sub_size, int bs, double regularizer, double lr, int r, int np, int ts): M(nn.W[0].n_cols), N(nn.W[1].n_rows), 
    K(nn.W[0].n_rows), num_sample(sub_size), batch_size(bs), reg(regularizer), learning_rate(lr), rank(r), num_procs(np), totalsize(ts) {



        stat = cublasCreate(&handle);
        if(stat != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "CUBLAS initialization failed!" << std::endl;
            return;
        }

        cudaMalloc((void**)&z0, sizeof(double) * K * num_sample);
        cudaMalloc((void**)&z1, sizeof(double) * N * num_sample);
        cudaMalloc((void**)&a0, sizeof(double) * K * num_sample);
        cudaMalloc((void**)&a1, sizeof(double) * N * num_sample);
        cudaMalloc((void**)&W0, sizeof(double) * M * K);
        cudaMalloc((void**)&W1, sizeof(double) * K * N);
        cudaMalloc((void**)&b0, sizeof(double) * K);
        cudaMalloc((void**)&b1, sizeof(double) * N);
        cudaMalloc((void**)&dexp, sizeof(double) * 1 * num_sample);
        cudaMalloc((void**)&dW0, sizeof(double) * M * K);
        cudaMalloc((void**)&dW1, sizeof(double) * K * N);
        cudaMalloc((void**)&db0, sizeof(double) * K);
        cudaMalloc((void**)&db1, sizeof(double) * N);
        cudaMalloc((void**)&dX, sizeof(double) * M * totalsize);
        cudaMalloc((void**)&dY, sizeof(double) * N * totalsize);


        dW0_h = (double*)malloc(sizeof(double)*M*K);
        dW1_h = (double*)malloc(sizeof(double)*K*N);
        db0_h = (double*)malloc(sizeof(double)*K);
        db1_h = (double*)malloc(sizeof(double)*N);

/*         for (unsigned int i=0; i<nn.W.size(); i++) {
            MPI_SAFE_CALL(MPI_Bcast(nn.W[i].memptr(), nn.W[i].n_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD));
            MPI_SAFE_CALL(MPI_Bcast(nn.b[i].memptr(), nn.b[i].n_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD));
        } */

        cudaMemcpy(b0, nn.b[0].memptr(), sizeof(double) * K , cudaMemcpyHostToDevice);
        cudaMemcpy(b1, nn.b[1].memptr(), sizeof(double) * N, cudaMemcpyHostToDevice);
        cudaMemcpy(W0, nn.W[0].memptr(), sizeof(double) * M * K, cudaMemcpyHostToDevice);
        cudaMemcpy(W1, nn.W[1].memptr(), sizeof(double) * K * N, cudaMemcpyHostToDevice);
        //std::cout<<totalsize<<" "<<M<<" "<<N<<" "<<K<<"\n";

/*      MPI_SAFE_CALL(MPI_Bcast(W0, M*K, MPI_DOUBLE, 0, MPI_COMM_WORLD));
        MPI_SAFE_CALL(MPI_Bcast(b0, K, MPI_DOUBLE, 0, MPI_COMM_WORLD));
        MPI_SAFE_CALL(MPI_Bcast(W1, K*N, MPI_DOUBLE, 0, MPI_COMM_WORLD));
        MPI_SAFE_CALL(MPI_Bcast(b1, N, MPI_DOUBLE, 0, MPI_COMM_WORLD)); */
        

    }


    void LoadData(const arma::mat& X, const arma::mat& y) {


        totalsize = (rank == 0)?X.n_cols:0;
        MPI_SAFE_CALL(MPI_Bcast(&totalsize, 1, MPI_INT, 0, MPI_COMM_WORLD));
        
        
        double* xdata=(double*)malloc(sizeof(double)*M*totalsize);
        double* ydata=(double*)malloc(sizeof(double)*totalsize*N);
        
        if (rank == 0) {
            cudaMemcpy(xdata, X.memptr(), sizeof(double) * M * totalsize, cudaMemcpyHostToHost);
            cudaMemcpy(ydata, y.memptr(), sizeof(double) * N * totalsize, cudaMemcpyHostToHost);
        }
       
        cudaMalloc((void**)&dX, sizeof(double) * M * totalsize);
        cudaMalloc((void**)&dY, sizeof(double) * N * totalsize);
        MPI_SAFE_CALL(MPI_Bcast(xdata, M*totalsize, MPI_DOUBLE, 0, MPI_COMM_WORLD));
        MPI_SAFE_CALL(MPI_Bcast(ydata, N*totalsize, MPI_DOUBLE, 0, MPI_COMM_WORLD));      

        //std::cout<<"X: \n"<<X.submat(0,0,5,5);
        cudaMemcpy(dX, xdata, sizeof(double) * M * totalsize, cudaMemcpyHostToDevice);
        cudaMemcpy(dY, ydata, sizeof(double) * N * totalsize, cudaMemcpyHostToDevice);
        free(xdata);
        free(ydata);



        }


    void LoadXY(int posx, int posy, const double* xptr, const double* yptr, int subsize) {
        
        cudaMemcpy(dX+posx, xptr, sizeof(double) * M * subsize, cudaMemcpyHostToDevice);
        cudaMemcpy(dY+posy, yptr, sizeof(double) * N * subsize, cudaMemcpyHostToDevice);

    }


    void FeedForward(int pos, int subsize, int wholesize)  {
        num_sample = subsize;
        batch_size = wholesize;

        gpu_repmat(b0, z0, K, num_sample);
        check_launch("repmat b0");
        gpu_repmat(b1, z1, N, num_sample);
        check_launch("repmat b1");
        
        




        double alpha = 1;
        double beta = 1;

        //gpu_addmat(dx, dX+pos, a1, 1, -1, M, num_sample);
        
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, num_sample, M, &alpha, W0, K, dX+pos, M, &beta, z0, K);
        check_launch("myGEMM 1");
        gpu_sigmoid(z0, a0, K, num_sample);
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, num_sample, K, &alpha, W1, N, a0, K, &beta, z1, N);
        check_launch("myGEMM 2");
        gpu_exp(z1, a1, N, num_sample);
        check_launch("exp");
        gpu_sumcol(a1, dexp, N, num_sample);
        check_launch("sumcol");
        gpu_softmax(dexp, a1, N, num_sample);

    } 

    void BackProp(int posx, int posy) {

        double alpha = 1/(double)(num_sample);
        double beta = -1/(double)(num_sample);
        double alpha1 = 1;
        double beta1=0;

        double r = reg/(double)(num_procs);
        
        //cudaMemcpy(dy, yptr, sizeof(double) * N * num_sample, cudaMemcpyHostToDevice);

        cudaMemcpy(dW0, W0, sizeof(double) * M * K, cudaMemcpyDeviceToDevice);
        cudaMemcpy(dW1, W1, sizeof(double) * K * N, cudaMemcpyDeviceToDevice);

        gpu_addmat(a1, dY+posy, a1, 1/(double)(batch_size), -1/(double)(batch_size), N, num_sample);
        check_launch("add mat");
        //myGEMM2(dW1, dDff, da1, &alpha1, &beta1, K, num_sample, N, true, false);
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, K, num_sample, N, &alpha1, W1, N, a1, N, &beta1, z0, K);

        check_launch("myGEMM");
        //myGEMM2(dDff, da0, dW1, &alpha1, &reg, N, K, num_sample, false, true);
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, K, num_sample, &alpha1, a1, N, a0, K, &r, dW1, N);
        check_launch("myGEMM 2");

        gpu_hadmard(z0, z0, a0, K, num_sample);
        check_launch("hadmard");


        gpu_sumrow(a1, db1, N, num_sample);
        check_launch("sumrow");

        //myGEMM2(dz1, dX, dW0, &alpha1, &reg, K, M, num_sample, false, true);
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, K, M, num_sample, &alpha1, z0, K, dX+posx, M, &r, dW0, K);
        check_launch("myGEMM 3");

        gpu_sumrow(z0, db0, K, num_sample);
        check_launch("sumrow2");



    } 

 




    void GradientDescent() {

        gpu_addmat(W0, dW0, W0, 1, -learning_rate, K, M);
        check_launch("addmat 1");
        gpu_addmat(W1, dW1, W1, 1, -learning_rate, N, K);
        check_launch("addmat 2");
        gpu_addmat(b0, db0, b0, 1, -learning_rate, K, 1);
        check_launch("addmat 3");
        gpu_addmat(b1, db1, b1, 1, -learning_rate, N, 1);
        check_launch("addmat 4");
    }

    void ReduceGradient() {

        cudaMemcpy(dW0_h, dW0, sizeof(double) * M * K, cudaMemcpyDeviceToHost);
        cudaMemcpy(db0_h, db0, sizeof(double) * K, cudaMemcpyDeviceToHost);
        cudaMemcpy(dW1_h, dW1, sizeof(double) * N * K, cudaMemcpyDeviceToHost);
        cudaMemcpy(db1_h, db1, sizeof(double) * N, cudaMemcpyDeviceToHost);

        MPI_SAFE_CALL(MPI_Allreduce(MPI_IN_PLACE, dW0_h, M * K, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
        MPI_SAFE_CALL(MPI_Allreduce(MPI_IN_PLACE, dW1_h, K * N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
        MPI_SAFE_CALL(MPI_Allreduce(MPI_IN_PLACE, db0_h, K, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
        MPI_SAFE_CALL(MPI_Allreduce(MPI_IN_PLACE, db1_h, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));

        cudaMemcpy(dW0, dW0_h, sizeof(double) * M * K, cudaMemcpyHostToDevice);
        cudaMemcpy(db0, db0_h, sizeof(double) * K, cudaMemcpyHostToDevice);
        cudaMemcpy(dW1, dW1_h, sizeof(double) * N * K, cudaMemcpyHostToDevice);
        cudaMemcpy(db1, db1_h, sizeof(double) * N, cudaMemcpyHostToDevice);
    }



    void UpdateCoefficient(NeuralNetwork& nn) {
        
        cudaMemcpy(nn.W[0].memptr(), W0, sizeof(double) * M * K, cudaMemcpyDeviceToHost);
        cudaMemcpy(nn.b[0].memptr(), b0, sizeof(double) * K, cudaMemcpyDeviceToHost);
        cudaMemcpy(nn.W[1].memptr(), W1, sizeof(double) * N * K, cudaMemcpyDeviceToHost);
        cudaMemcpy(nn.b[1].memptr(), b1, sizeof(double) * N, cudaMemcpyDeviceToHost);

    }

    int T1() {return totalsize;}
    int M1() {return M;}
    int K1() {return K;}
    int N1() {return N;}


    ~OneBatchUpdate2()   {
        
        cudaFree(W0);
        cudaFree(W1);
        cudaFree(b0);
        cudaFree(b1);
        cudaFree(z0);
        cudaFree(z1);
        cudaFree(a0);
        cudaFree(a1);
        cudaFree(dW0);
        cudaFree(dW1);
        cudaFree(db0);
        cudaFree(db1);
        cudaFree(dX);
        cudaFree(dY);
        cudaFree(dexp);

    }

    


    private:

    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;

    int totalsize;
    int rank;
    int num_sample;
    int batch_size;
    int N;
    int M;
    int K;

    double* z0; 
    double* z1;
    double* a0;
    double* a1;
    double* W0;
    double* W1;
    double* b0;
    double* b1;
    double* dW0;
    double* dW1;
    double* db0;
    double* db1;
    double* dX;
    double* dexp;
    double* dY;
    double reg;
    double learning_rate;
    double* dW0_h;
    double* dW1_h;
    double* db0_h;
    double* db1_h;


    int num_procs;




};


class OneBatchUpdate  {



    public:
    OneBatchUpdate(NeuralNetwork& nn, int sub_size, int total_size, double regularizer, double lr, int r, int np): M(nn.W[0].n_cols), N(nn.W[1].n_rows), 
    K(nn.W[0].n_rows), num_sample(sub_size), batch_size(total_size), reg(regularizer), learning_rate(lr), rank(r), num_procs(np) {


        stat = cublasCreate(&handle);
        if(stat != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "CUBLAS initialization failed!" << std::endl;
            return;
        }

        cudaMalloc((void**)&z0, sizeof(double) * K * num_sample);
        cudaMalloc((void**)&z1, sizeof(double) * N * num_sample);
        cudaMalloc((void**)&a0, sizeof(double) * K * num_sample);
        cudaMalloc((void**)&a1, sizeof(double) * N * num_sample);
        cudaMalloc((void**)&W0, sizeof(double) * M * K);
        cudaMalloc((void**)&W1, sizeof(double) * K * N);
        cudaMalloc((void**)&b0, sizeof(double) * K);
        cudaMalloc((void**)&b1, sizeof(double) * N);
        cudaMalloc((void**)&dX, sizeof(double) * M * num_sample);
        cudaMalloc((void**)&dexp, sizeof(double) * 1 * num_sample);
        cudaMalloc((void**)&dW0, sizeof(double) * M * K);
        cudaMalloc((void**)&dW1, sizeof(double) * K * N);
        cudaMalloc((void**)&db0, sizeof(double) * K);
        cudaMalloc((void**)&db1, sizeof(double) * N);
        cudaMalloc((void**)&dyc, sizeof(double) * N * num_sample);
        cudaMalloc((void**)&dy, sizeof(double) * N * num_sample);
        dW0_h = (double*)malloc(sizeof(double)*M*K);
        dW1_h = (double*)malloc(sizeof(double)*K*N);
        db0_h = (double*)malloc(sizeof(double)*K);
        db1_h = (double*)malloc(sizeof(double)*N);


        cudaMemcpy(b0, nn.b[0].memptr(), sizeof(double) * K , cudaMemcpyHostToDevice);
        cudaMemcpy(b1, nn.b[1].memptr(), sizeof(double) * N, cudaMemcpyHostToDevice);
        cudaMemcpy(W0, nn.W[0].memptr(), sizeof(double) * M * K, cudaMemcpyHostToDevice);
        cudaMemcpy(W1, nn.W[1].memptr(), sizeof(double) * K * N, cudaMemcpyHostToDevice);


/*      MPI_SAFE_CALL(MPI_Bcast(W0, M*K, MPI_DOUBLE, 0, MPI_COMM_WORLD));
        MPI_SAFE_CALL(MPI_Bcast(b0, K, MPI_DOUBLE, 0, MPI_COMM_WORLD));
        MPI_SAFE_CALL(MPI_Bcast(W1, K*N, MPI_DOUBLE, 0, MPI_COMM_WORLD));
        MPI_SAFE_CALL(MPI_Bcast(b1, N, MPI_DOUBLE, 0, MPI_COMM_WORLD)); */
        

    }




    void FeedForward(const double* xptr, int subsize, int wholesize)  {
        num_sample = subsize;
        batch_size = wholesize;
        cudaMemcpy(dX, xptr, sizeof(double) * M * num_sample, cudaMemcpyHostToDevice);
        check_launch("Copying X");
        gpu_repmat(b0, z0, K, num_sample);
        check_launch("repmat b0");
        gpu_repmat(b1, z1, N, num_sample);
        check_launch("repmat b1");
        




        double alpha = 1;
        double beta = 1;

        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, num_sample, M, &alpha, W0, K, dX, M, &beta, z0, K);
        check_launch("myGEMM 1");
        gpu_sigmoid(z0, a0, K, num_sample);
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, num_sample, K, &alpha, W1, N, a0, K, &beta, z1, N);
        check_launch("myGEMM 2");
        gpu_exp(z1, a1, N, num_sample);
        check_launch("exp");
        gpu_sumcol(a1, dexp, N, num_sample);
        check_launch("sumcol");
        gpu_softmax(dexp, a1, N, num_sample);

    }

    void BackProp(const double* yptr) {

        double alpha = 1/(double)(num_sample);
        double beta = -1/(double)(num_sample);
        double alpha1 = 1;
        double beta1=0;

        double r = reg/(double)(num_procs);
        
        cudaMemcpy(dy, yptr, sizeof(double) * N * num_sample, cudaMemcpyHostToDevice);
        cudaMemcpy(dW0, W0, sizeof(double) * M * K, cudaMemcpyDeviceToDevice);
        cudaMemcpy(dW1, W1, sizeof(double) * K * N, cudaMemcpyDeviceToDevice);

        gpu_addmat(a1, dy, dy, 1/(double)(batch_size), -1/(double)(batch_size), N, num_sample);
        check_launch("add mat");
        //myGEMM2(dW1, dDff, da1, &alpha1, &beta1, K, num_sample, N, true, false);
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, K, num_sample, N, &alpha1, W1, N, dy, N, &beta1, z0, K);
        check_launch("myGEMM");
        //myGEMM2(dDff, da0, dW1, &alpha1, &reg, N, K, num_sample, false, true);
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, K, num_sample, &alpha1, dy, N, a0, K, &r, dW1, N);
        check_launch("myGEMM 2");

        gpu_hadmard(z0, z0, a0, K, num_sample);
        check_launch("hadmard");


        gpu_sumrow(dy, db1, N, num_sample);
        check_launch("sumrow");

        //myGEMM2(dz1, dX, dW0, &alpha1, &reg, K, M, num_sample, false, true);
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, K, M, num_sample, &alpha1, z0, K, dX, M, &r, dW0, K);
        check_launch("myGEMM 3");

        gpu_sumrow(z0, db0, K, num_sample);
        check_launch("sumrow2");



    }


    void GradientDescent() {

        gpu_addmat(W0, dW0, W0, 1, -learning_rate, K, M);
        check_launch("addmat 1");
        gpu_addmat(W1, dW1, W1, 1, -learning_rate, N, K);
        check_launch("addmat 2");
        gpu_addmat(b0, db0, b0, 1, -learning_rate, K, 1);
        check_launch("addmat 3");
        gpu_addmat(b1, db1, b1, 1, -learning_rate, N, 1);
        check_launch("addmat 4");
    }

    void ReduceGradient() {

        cudaMemcpy(dW0_h, dW0, sizeof(double) * M * K, cudaMemcpyDeviceToHost);
        cudaMemcpy(db0_h, db0, sizeof(double) * K, cudaMemcpyDeviceToHost);
        cudaMemcpy(dW1_h, dW1, sizeof(double) * N * K, cudaMemcpyDeviceToHost);
        cudaMemcpy(db1_h, db1, sizeof(double) * N, cudaMemcpyDeviceToHost);

        MPI_SAFE_CALL(MPI_Allreduce(MPI_IN_PLACE, dW0_h, M * K, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
        MPI_SAFE_CALL(MPI_Allreduce(MPI_IN_PLACE, dW1_h, K * N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
        MPI_SAFE_CALL(MPI_Allreduce(MPI_IN_PLACE, db0_h, K, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
        MPI_SAFE_CALL(MPI_Allreduce(MPI_IN_PLACE, db1_h, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));

        cudaMemcpy(dW0, dW0_h, sizeof(double) * M * K, cudaMemcpyHostToDevice);
        cudaMemcpy(db0, db0_h, sizeof(double) * K, cudaMemcpyHostToDevice);
        cudaMemcpy(dW1, dW1_h, sizeof(double) * N * K, cudaMemcpyHostToDevice);
        cudaMemcpy(db1, db1_h, sizeof(double) * N, cudaMemcpyHostToDevice);
    }



    void UpdateCoefficient(NeuralNetwork& nn) {
        
        cudaMemcpy(nn.W[0].memptr(), W0, sizeof(double) * M * K, cudaMemcpyDeviceToHost);
        cudaMemcpy(nn.b[0].memptr(), b0, sizeof(double) * K, cudaMemcpyDeviceToHost);
        cudaMemcpy(nn.W[1].memptr(), W1, sizeof(double) * N * K, cudaMemcpyDeviceToHost);
        cudaMemcpy(nn.b[1].memptr(), b1, sizeof(double) * N, cudaMemcpyDeviceToHost);


    }



    ~OneBatchUpdate()   {
        
        cudaFree(W0);
        cudaFree(W1);
        cudaFree(b0);
        cudaFree(b1);
        cudaFree(z0);
        cudaFree(z1);
        cudaFree(a0);
        cudaFree(a1);
        cudaFree(dW0);
        cudaFree(dW1);
        cudaFree(db0);
        cudaFree(db1);
        cudaFree(dX);
        cudaFree(dexp);
        cudaFree(dyc);
        cudaFree(dy);
    }

    


    private:

    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;

    int rank;
    int num_sample;
    int batch_size;
    int N;
    int M;
    int K;

    double* z0; 
    double* z1;
    double* a0;
    double* a1;
    double* W0;
    double* W1;
    double* b0;
    double* b1;
    double* dW0;
    double* dW1;
    double* db0;
    double* db1;
    double* dX;
    double* dexp;
    double* dy;
    double* dyc;
    double reg;
    double learning_rate;
    double* dW0_h;
    double* dW1_h;
    double* db0_h;
    double* db1_h;


    int num_procs;




};








void parallel_train(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
                    double learning_rate, double reg,
                    const int epochs, const int batch_size, bool grad_check, int print_every,
                    int debug) {

    int rank, num_procs;
    MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
    MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    std::ofstream error_file;
    error_file.open("Outputs/CpuGpuDiff.txt");
    int print_flag = 0;
    int iter = 0;
    int N = (rank == 0)?X.n_cols:0;
    MPI_SAFE_CALL(MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD));

    int *displsx = new int[num_procs];
    int *displsy = new int[num_procs];
    int *countsx = new int[num_procs];
    int *countsy = new int[num_procs];
    int x_row = X.n_rows;
    int y_row = y.n_rows;
    MPI_SAFE_CALL(MPI_Bcast(&x_row, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_SAFE_CALL(MPI_Bcast(&y_row, 1, MPI_INT, 0, MPI_COMM_WORLD));
    int subsize = (batch_size + num_procs - 1) / num_procs;
    int this_batch_size = batch_size;
    double* xptr_sub = (double*)malloc(sizeof(double)*x_row*subsize);
    double* yptr_sub = (double*)malloc(sizeof(double)*y_row*subsize);

    OneBatchUpdate2 pp(nn, subsize, batch_size, reg, learning_rate, rank, num_procs, N);
    for(int epoch = 0; epoch < epochs; ++epoch) {
        int num_batches = (N + batch_size - 1)/batch_size;
        for(int batch = 0; batch < num_batches; ++batch) {


            int last_col = std::min((batch + 1) * batch_size-1, N-1);
            this_batch_size = last_col - batch * batch_size + 1;
            subsize = (this_batch_size + num_procs - 1) / num_procs;
            int counts = (rank == (num_procs - 1)) ? (this_batch_size-(num_procs-1)*subsize) : subsize;
            std::cout<<rank<<" "<<counts<<"\n";
            
            int xpos = batch * batch_size * x_row + subsize * rank * x_row;
            int ypos = batch * batch_size * y_row + subsize * rank * y_row;

            if (epoch == 0) {
                const double* xptr = X.memptr() + batch * batch_size * x_row;
                const double* yptr = y.memptr() + batch * batch_size * y_row;
                for (unsigned int i = 0; i < num_procs; i++) {
                    displsx[i] = subsize * i * x_row;
                    countsx[i] = (rank == (num_procs - 1)) ? ((this_batch_size-(num_procs-1)*subsize) * x_row) : subsize * x_row;
                    displsy[i] = subsize * i * y_row;
                    countsy[i] = (rank == (num_procs - 1)) ? ((this_batch_size-(num_procs-1)*subsize) * y_row) : subsize * y_row;
                }
                if (rank == 0 ) {
                    for (unsigned int i = 0; i < num_procs; i++) {
                    printf("displsx[%d]=%d\n", i, displsx[i]);
                    printf("countsx[%d]=%d\n", i, countsx[i]);
                    printf("displsy[%d]=%d\n", i, displsy[i]);
                    printf("countsy[%d]=%d\n", i, countsy[i]);
                }
                }
                MPI_SAFE_CALL(MPI_Scatterv(xptr, countsx, displsx, MPI_DOUBLE, xptr_sub, countsx[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD));
                MPI_SAFE_CALL(MPI_Scatterv(yptr, countsy, displsy, MPI_DOUBLE, yptr_sub, countsy[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD)); 
                pp.LoadXY(xpos, ypos, xptr_sub, yptr_sub, counts);

            }

            pp.FeedForward(xpos, counts, this_batch_size);
            pp.BackProp(xpos, ypos);
            pp.ReduceGradient();
            pp.GradientDescent();
            if(debug && rank == 0 && print_flag) {
                write_diff_gpu_cpu(nn, iter, error_file);
            }
/*             if (iter > 5) {
                return;
            } */
            iter++;

        }
    }
    pp.UpdateCoefficient(nn);
    free(xptr_sub);
    free(yptr_sub);
    error_file.close();
}

void parallel_train2(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
                    double learning_rate, double reg,
                    const int epochs, const int batch_size, bool grad_check, int print_every,
                    int debug) {

    int rank, num_procs;
    MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
    MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    int N = (rank == 0)?X.n_cols:0;
    MPI_SAFE_CALL(MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD));
    int x_row = X.n_rows;
    int y_row = y.n_rows;
    MPI_SAFE_CALL(MPI_Bcast(&x_row, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_SAFE_CALL(MPI_Bcast(&y_row, 1, MPI_INT, 0, MPI_COMM_WORLD));
    std::ofstream error_file;
    error_file.open("Outputs/CpuGpuDiff.txt");
    int print_flag = 0;
    int iter = 0;
    int *displsx = new int[num_procs];
    int *displsy = new int[num_procs];
    int *countsx = new int[num_procs];
    int *countsy = new int[num_procs];
    int subsize = (batch_size + num_procs - 1) / num_procs;
    int this_batch_size = batch_size;
    double* xptr_sub = (double*)malloc(sizeof(double)*x_row*subsize);
    double* yptr_sub = (double*)malloc(sizeof(double)*y_row*subsize);
    for (unsigned int i=0; i<nn.W.size(); i++) {
        MPI_SAFE_CALL(MPI_Bcast(nn.W[i].memptr(), nn.W[i].n_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD));
        MPI_SAFE_CALL(MPI_Bcast(nn.b[i].memptr(), nn.b[i].n_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD));
    }
    //std::cout<<"Broadcast done...\n";
    OneBatchUpdate pp(nn, subsize, batch_size, reg, learning_rate, rank, num_procs);
    //std::cout<<"Initialization done...\n";
    for(int epoch = 0; epoch < epochs; ++epoch) {
        int num_batches = (N + batch_size - 1)/batch_size;
        for(int batch = 0; batch < num_batches; ++batch) {
            //std::cout<<"Calculating pointer...\n";
            const double* xptr = X.memptr() + batch * batch_size * x_row;
            const double* yptr = y.memptr() + batch * batch_size * y_row;

            int last_col = std::min((batch + 1) * batch_size-1, N-1);
            this_batch_size = last_col - batch * batch_size + 1;
            subsize = (this_batch_size + num_procs - 1) / num_procs;
            //std::cout<<"Assigning positions...\n";
            for (unsigned int i = 0; i < num_procs; i++) {
                displsx[i] = subsize * i * x_row;
                countsx[i] = subsize * x_row;
                displsy[i] = subsize * i * y_row;
                countsy[i] = subsize * y_row;
            }

            //std::cout<<rank<<" rank Scatter begins...\n";
            MPI_SAFE_CALL(MPI_Scatterv(xptr, countsx, displsx, MPI_DOUBLE, xptr_sub, countsx[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD));
            MPI_SAFE_CALL(MPI_Scatterv(yptr, countsy, displsy, MPI_DOUBLE, yptr_sub, countsy[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD)); 
            //std::cout<<rank<<" rank Scatter done...\n";
            pp.FeedForward(xptr_sub, subsize, this_batch_size);
            //std::cout<<rank<<"Feedforward done...\n";
            pp.BackProp(yptr_sub);
            //std::cout<<rank<<"Backprop done...\n";
            pp.ReduceGradient();
            //std::cout<<"Reduce done...\n";
            pp.GradientDescent();
            if(debug && rank == 0 && print_flag) {
                write_diff_gpu_cpu(nn, iter, error_file);
            }

            iter++;

        }
    }
    pp.UpdateCoefficient(nn);
    free(xptr_sub);
    free(yptr_sub);
    error_file.close();
}