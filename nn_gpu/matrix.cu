/* Copyright 2019, Aman Gupta, ENG EC 527, Prof. Martin Herbordt
/******************************************************************************/
/* Matrix library implementation for CUDA in C
*/
#pragma once    /* File Guard */
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
    // small check to ensure rows and cols are positive numbers
    if (rows <= 0 || cols <= 0) return NULL;
    int N = rows*cols;

    // memory allocation of matrix
    matrix *m = (matrix *) malloc(sizeof(matrix));
    data_t *mat_data = (data_t *)calloc(N,sieof(data_t));
    // check if continuous memory allocation for data failed
    if (!(mat_data)) return NULL;
    
    m -> rows = rows;
    m -> cols = cols;
    m -> data = mat_data;

    // initializate matrix data with random values
    int i; srand(seed);
    for (i = 0; i < N; i++)
        m -> data[i] = (float)rand();

    return m;
}

/* Matrix initialization on Host
Input:  Matrix
Output: 0 on successfull deletion of matrix, -1 otherwise
*/
__host__
int matrix_delete(matrix *mat) {
    if (!mat) return -1;
    assert(mat -> data);
    free(mat -> data);
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
            ELEMENT(prod, row, col) = val;
        }
    return 0;
}

/* Matrix Multiplication kernel with global memory
Input: Matrix A, B, out
Output: A*B stored in out and 0 on success. returns -1 on failure
        Assumes inner dimensions match, returns -2 if they dont match
        Assumes global memory view
*/
__global__
void MMM_global(matrix *A, matrix *B, matrix *out) {

}

/* Matrix Multiplication kernel with shared memory
Input: Matrix A, B, out
Output: A*B stored in out and 0 on success. returns -1 on failure
        Assumes inner dimensions match, returns -2 if they dont match
        Assumes shared memory view
*/
__global__
void MMM_shared(matrix *A, matrix *B, matrix *out) {

}

/* Matrix Multiplication kernel with unified memory
Input: Matrix A, B, out
Output: A*B stored in out and 0 on success. returns -1 on failure
        Assumes inner dimensions match, returns -2 if they dont match
        Assumes unified memory view
*/
__global__
void MMM_unified(matrix *A, matrix *B, matrix *out) {
    
}
