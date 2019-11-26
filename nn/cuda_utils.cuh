#pragma once /* File Guard */

#include "matrix.cuh"

// Assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), (char *)__FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "CUDA_SAFE_CALL: %s %s %d\n",
                                       cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

void matrix_allocate_cuda(Matrix *A);

int matrix_free_cuda(Matrix *A);

void copy_matrix_H2D(Matrix *A);

void copy_matrix_D2H(Matrix *A);