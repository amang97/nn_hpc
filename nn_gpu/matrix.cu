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
