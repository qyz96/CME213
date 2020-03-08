/**
 * @file util.h
 * @brief Provides utility functions for the DNS homework.
 * 
 * Do not modify this file. It will not be turned in anyway.
 */

#ifndef _UTIL_H_
#define _UTIL_H_

#include <cmath>
#include <iostream>

/**
 * @brief A * B -> C
 * Multiplies square matrices A and B and places them into C.
 * 
 * @param[in]  A   Input square matrix
 * @param[in]  B   Input square matrix
 * @param[out] C   Output square matrix
 * @param[in]  n   Square matrix dimension
 * @pre A, B, C != nullptr
 * @tparam T Matrix data type
 */
template <typename T>
void naive_matmul(const T *A, const T *B, T *C, std::size_t n)
{
  for (std::size_t i = 0; i < n; i++)
  {
    for (std::size_t j = 0; j < n; j++)
    {
      C[i * n + j] = 0;
      for (std::size_t k = 0; k < n; k++)
      {
          C[i * n + j] += A[i * n + k] * B[k * n + j];
      }
    }
  }
}

/**
 * @brief A * B -> C, but this time with OpenMP.
 * Multiplies square matrices A and B and places them into C. Uses OpenMP to
 * parallelize outer loop.
 * 
 * @param[in]  A   Input square matrix
 * @param[in]  B   Input square matrix
 * @param[out] C   Output square matrix
 * @param[in]  n   Square matrix dimension
 * @pre A, B, C != nullptr
 * @tparam T Matrix data type
 */
template <typename T>
void omp_matmul(const T *A, const T *B, T *C, std::size_t n)
{
  #pragma omp parallel for
  for (std::size_t i = 0; i < n; i++)
  {
    for (std::size_t j = 0; j < n; j++)
    {
      C[i * n + j] = 0;
      for (std::size_t k = 0; k < n; k++)
      {
          C[i * n + j] += A[i * n + k] * B[k * n + j];
      }
    }
  }
}

/**
 * Computes the L2 norm between two vectors, or an elementwise L2 norm on two
 * row-major matrices.
 * 
 * @param[in] A   input array to compare
 * @param[in] B   input array to compare
 * @param[in] n   number of elements in A and Bs
 * @return ||A - B||_2
 */
double l2_norm(const float *A, const float *B, std::size_t n)
{
  double sq_dev = 0;
  for (std::size_t i = 0; i < n; i++)
    sq_dev += (A[i] - B[i]) * (A[i] - B[i]);
  return sqrt(sq_dev);
}

/**
 * Prints a square matrix @a a with dimensions @a n x @a n.
 * @param[in] a   matrix to print
 * @param[in] n   matrix dimension 
 * @tparam T      matrix data type
 */
template <typename T>
void print_sq_matrix(const T *a, std::size_t n)
{
    for (std::size_t i = 0; i < n; i++)
    {
      for (std::size_t j = 0; j < n; j++)
      {
        std::cout << a[i * n + j] << " ";
      }
      std::cout << std::endl;
    }
}

union Float_t
{
    Float_t(float num) : f(num) {}
    bool Negative() const { return (i >> 31) != 0; }

    int32_t i;
    float f;
};


// ref: https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
bool AlmostEqualUlps(float A, float B, int maxUlpsDiff) 
{
    Float_t uA(A);
    Float_t uB(B);

    // Different signs means they do not match.
    if (uA.Negative() != uB.Negative()) 
    {
        // Check for equality to make sure +0 == -0
        if (A == B)
            return true;
        return false;
    }

    // Find the difference in ULPs.
    int ulpsDiff = abs(uA.i - uB.i);

    if (ulpsDiff <= maxUlpsDiff)
        return true;
    return false;
}

#endif
