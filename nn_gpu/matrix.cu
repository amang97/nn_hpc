/* Copyright 2019, Aman Gupta, ENG EC 527, Prof. Martin Herbordt              */
/******************************************************************************/
/* Matrix library implementation for CUDA in C                                */
/******************************************************************************/
/* Libraries */
/******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "matrix.cuh"
/******************************************************************************/
/* Implementations */
/******************************************************************************/
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

/* Matrix Allocation on host
    Input: num of rows, num of cols
    Output: allocated memory pointer to matrix type
*/
__host__
matrix * matrix_allocate(int rows, int cols) {
    // small check to ensure rows and cols are positive numbers
    if (rows <= 0 || cols <= 0) return NULL;
    int N = rows*cols;

    // memory allocation of matrix
    matrix *m = (matrix *) malloc(sizeof(matrix));
    data_t *mat_data = (data_t *)calloc(N,sizeof(data_t));
    // check if continuous memory allocation for data failed
    if (!(mat_data)) return NULL;

    // Allocate host memory
    m -> rows = rows;
    m -> cols = cols;
    m -> data_h = mat_data;
    // Allocate device memory
    CUDA_SAFE_CALL(cudaMalloc(&(m->data_d), N*sizeof(data_t)));
    return m;
}

/* Matrix initialization on Host
Input:  number of rows and columns. Initialization seed for filling the matrix
        and reproducability
Output: NULL in case the rows or column are not positive, or failure in memory
        allocation
        on success, a pointer to matrix type of size rows*cols with randomly
        initialized data of type data_t
*/
__host__
matrix * matrix_init(int rows, int cols, int seed) {
    matrix *m = matrix_allocate(rows,cols);
    // initializate matrix data with random values if seed on host
    int i;
    int N = rows*cols;
    if (seed) {
        srand(seed);
        for (i = 0; i < N; i++)
            m -> data_h[i] = (data_t)rand();
    } else for (i = 0; i < N; i++) m -> data_h[i] = (data_t)0;
    // copy host data to device
    CUDA_SAFE_CALL(cudaMemcpy(m->data_d, m->data_h, N*sizeof(data_t),
                                                    cudaMemcpyHostToDevice));
    return m;
}

/* Matrix initialization on Host
Input:  Matrix
Output: 0 on successfull deletion of matrix, -1 otherwise
*/
__host__
int matrix_delete(matrix *mat) {
    if (!mat) return -1;
    assert(mat -> data_h); free(mat -> data_h);
    CUDA_SAFE_CALL(cudaFree(mat -> data_d));
    free(mat);
    return 0;
}

/* Matrix Multiplication on host
Input: Matrix A, B, out
Output: A*B stored in out and 0 on success. returns -1 on failure
        Assumes inner dimensions match, returns -2 if they dont match
        Assumes global memory view
*/
__host__
int MMM(matrix *A, matrix *B, matrix *out) {
    if ((!A) || (!B) || (!out)) return -1;
    if ((A->cols != B->rows) ||
        (A->rows != out->rows) ||
        (B->cols != out->cols)) return -2;

    int row, col, k;
    for (col = 1; col <= B->cols; col++)
        for (row = 1; row <= A->rows; row++) {
            data_t val = (data_t)0;
            for (k = 1; k <= A->cols; k++)
                val += ELEMENT(A, row, k) * ELEMENT(B, k, col);
            ELEMENT(out, row, col) = val;
        }
    return 0;
}

/*
__host__
data_t * copy_matrix(data_t *A, data_t *B, int Bx, int By) {
  int N =Bx*By;
  data_t * cp = (data_t *)calloc(N,sizeof(data_t));
  memcpy(cp, B, N*sizeof(data_t));
  return cp;
}
*/
