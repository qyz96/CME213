/**
 * @file serialmmm.cpp
 * @brief Provides benchmark timings for non-MPI matrix multiplication.
 * 
 * It would make sense to place these benchmarks with dnsmm, but the timings
 * for the serial matrix multiplication were significantly slower when in the
 * MPI environment. Thus, the serial and OpenMP timings have been spun off into
 * a different executable.
 * 
 * Do not modify this file. It will not be turned in anyway.
 */

#include <chrono>
#include <random>
#include <vector>
#include <unistd.h>

#include "util.h"
#include "omp.h"

using namespace std::chrono;

constexpr std::size_t MATRIX_SIZE_START = 576;
constexpr std::size_t MATRIX_SIZE_END = 2880;
constexpr std::size_t MATRIX_SIZE_STRIDE = 96;

std::default_random_engine re;
std::uniform_real_distribution<float> u(0.0f, 1.0f);

void array_fill_random(float *a, float *b, std::size_t elements)
{
  #pragma omp parallel for
  for (std::size_t i = 0; i < elements; i++)
  {
    a[i] = u(re);
    b[i] = u(re);
  }
}

void print_usage(const char *program_name)
{
  fprintf(stderr,
    "Usage: %s [-so]\n"
    "-s\tRun naive matrix multiplication\n"
    "-o\tRun OpenMP matrix multiplication\n"
    "Both options can be passed\n", program_name
  );
}

int main(int argc, char **argv)
{
  bool run_naive = false;
  bool run_omp = false;

  if (argc == 1)
  {
    print_usage(argv[0]);
    return 1;
  }

  int opt;
  while ((opt = getopt(argc, argv, "so")) != -1)
  {
    switch (opt)
    {
      case 's':
        run_naive = true;
        break;
      case 'o':
        run_omp = true;
        break;
      default:
        print_usage(argv[0]);
        return 1;
    }
  }

  if (run_omp)
    printf("OpenMP is running with %d threads.\n", omp_get_max_threads());

  // print header
  printf("%-16s", "Matrix Dim");
  if (run_naive)
    printf("%-12s", "Naive");
  if (run_omp)
    printf("%-12s", "OpenMP");
  printf("\n");

  for (std::size_t n = MATRIX_SIZE_START; n <= MATRIX_SIZE_END; n += MATRIX_SIZE_STRIDE)
  {
    float *a = new float[n * n];
    float *b = new float[n * n];
    float *c = new float[n * n];

    array_fill_random(a, b, n * n);

    printf("%-16lu", n);

    if (run_naive)
    {
      auto start = steady_clock::now();
      naive_matmul(a, b, c, n);
      auto end = steady_clock::now();
      double elapsed_seconds = duration_cast<duration<double>>(end - start).count();    
      printf("%-12f", elapsed_seconds);
    }

    if (run_omp)
    {
      auto start = steady_clock::now();
      omp_matmul(a, b, c, n);
      auto end = steady_clock::now();
      double elapsed_seconds = duration_cast<duration<double>>(end - start).count();
      printf("%-12f", elapsed_seconds);
    }

    printf("\n");

    delete[] a;
    delete[] b;
    delete[] c;
  }

  return 0;
}
