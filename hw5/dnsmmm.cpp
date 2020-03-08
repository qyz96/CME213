/**
 * @file dnsmmm.cpp
 * @brief Entry point to run the DNS multiplication algorithm
 * 
 * In this homework, you are asked to implement the Dekel-Nassimi-Sahni (DNS)
 * matrix multiplication algorithm. This implementation follows the algorithm
 * described in the book "Introduction to Parallel Computing" by Grama, et. al
 * in Chapter 8.2.3 and 8.2.4. We will be implementing the block version 
 * described in 8.2.4.
 * 
 * The DNS algorithm assumes that the matrices A and B are already
 * evenly distributed across all ranks in the k = 0 plane. Instead, we will
 * assume that the root rank (rank 0) will have A and B. When running the DNS
 * algorithm, we would like to distribute necessary data to the appropriate
 * ranks as described in the DNS algorithm. Once the computation is complete, 
 * we would like rank 0 to have the output stored into C.
 * 
 * Your job is to implement the scattering of A and B to the other ranks,
 * the DNS algorithm itself, and the gathering of the sub-outputs back to the 
 * root rank.
 * 
 * Do not modify this file. It will not be turned in.
 */

#include <cstddef>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <random>
#include <mpi.h>

#include "util.h"
#include "dns.h"

/** Max number of floating point differences to tolerate */
constexpr int MAX_ULPS_DIFF = 512;

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int num_procs, my_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  int q = static_cast<int>(cbrt(num_procs));
  if (q * q * q != num_procs)
  {
    if (my_rank == 0)
    {
      std::cerr << "Number of processes must be a perfect cube." << std::endl;
      std::cerr << "usage: mpirun -np <# procs> " << argv[0] << " <matrix size>"
                << std::endl;
    }
    MPI_Finalize();
    exit(1);
  }

  if (argc != 2)
  {
    if (my_rank == 0)
    {
      std::cerr << "Must supply matrix size." << std::endl;
      std::cerr << "usage: mpirun -np <# procs> " << argv[0] << " <matrix size>"
                << std::endl;
    }
    MPI_Finalize();
    exit(1);
  }

  // compute the block size for each rank
  int matrix_dim = std::atoi(argv[1]);
  int blockdim = matrix_dim / q;
  if (matrix_dim < 0 || blockdim * q != matrix_dim)
  {
    if (my_rank == 0)
    {
      std::cerr << "The matrix dimension must be a multiple of cbrt(# procs)"
                << std::endl;
    }
    MPI_Finalize();
    exit(1);
  }

  if (my_rank == 0)
  {
    printf("Matrix dimension: %dx%d\n", matrix_dim, matrix_dim);
    printf("Topology: %dx%dx%d for a total of %d procs.\n", q, q, q, num_procs);
    printf("Each rank has block size: %dx%d\n", blockdim, blockdim);
  }

  float *a = nullptr;
  float *b = nullptr;
  float *c = nullptr;
  float *c_gold = nullptr;

  // rank 0 will generate A and B, and store final output C
  // generate A and B, and allocate storage for output, compute gold values
  if (my_rank == 0)
  {
    a = new float[matrix_dim * matrix_dim];
    b = new float[matrix_dim * matrix_dim];
    c = new float[matrix_dim * matrix_dim];
    c_gold = new float[matrix_dim * matrix_dim];

    // generate A & b
    std::minstd_rand rand_gen(2020);
    const int rand_interval = (1 << 4);
    for (int i = 0; i < matrix_dim; i++)
    {
      for (int j = 0; j < matrix_dim; j++)
      {
        // Option 1
        // This generates pseudo-random numbers with few non-zero bits.
        // This avoids roundoff errors during the multiplication.
        // % rand_interval     this ensures that the integers are small so that roundoff errors are avoided
        a[i * matrix_dim + j] = float(rand_gen() % rand_interval) - (float(rand_interval) - 1.0) / 2.0;
        b[i * matrix_dim + j] = a[i * matrix_dim + j];
        // Option 2
        // a[i * matrix_dim + j] = float(i * matrix_dim + j) / (matrix_dim * matrix_dim);
        // b[i * matrix_dim + j] = float(i * matrix_dim + j) / (matrix_dim * matrix_dim);
      }
    }
#ifndef TIMING
    std::cout << "Running serial matrix multiply for correctness..."
              << std::flush;
    omp_matmul(a, b, c_gold, matrix_dim);
    std::cout << "done." << std::endl;
#else
    std::cout << "Skipping serial matrix multiply & verification as timing enabled."
              << std::endl;
#endif
    std::cout << "Running parallel DNS..." << std::flush;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // set the stage: create topology
  struct mesh_info mesh_info = initialize_topology(blockdim, q);

  // run DNS
  double parallel_start = MPI_Wtime();
  dns_multiply(mesh_info, a, b, c, q, matrix_dim);
  double parallel_end = MPI_Wtime();

  // report timing & verify results
  if (my_rank == 0)
  {
    double parallel_time = parallel_end - parallel_start;
    std::cout << "done." << std::endl;
    std::cout << "DNS took " << parallel_time << " sec." << std::endl;
#ifndef TIMING
    std::cout << "Verifying..." << std::endl;
    int errors = 0;

    for (int i = 0; i < matrix_dim * matrix_dim; i++)
    {
      if (!AlmostEqualUlps(c[i], c_gold[i], MAX_ULPS_DIFF))
        errors++;
    }

    if (errors)
      std::cerr << "There were " << errors << " differences found between DNS and the serial algorithm."
                << std::endl;
    else
      std::cerr << "MPI output matched serial algorithm." << std::endl;

    std::cout << "L2 error between DNS and serial: "
              << l2_norm(c, c_gold, matrix_dim) << std::endl;
#endif
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // cleanup
  mesh_info_free(mesh_info);
  delete[] a;
  delete[] b;
  delete[] c;
  delete[] c_gold;
  MPI_Finalize();
  return MPI_SUCCESS;
}
