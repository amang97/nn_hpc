/* Copyright 2019, Aman Gupta, ENG EC 527, Prof. Martin Herbordt              */
/******************************************************************************/
/* Matrix library for GPU in C */
#pragma once    /* file guard */
/******************************************************************************/
/* Data Structures */
/******************************************************************************/
typedef float data_t;

typedef struct Matrix {
    int rows;
    int cols;
    data_t * data_d;
    data_t * data_h;
} matrix;

/******************************************************************************/
/* prototypes and usage */
/******************************************************************************/
/* Matrix Allocation on host
    Input: num of rows, num of cols
    Output: allocated memory pointer to matrix type
*/
__host__
matrix * matrix_allocate(int rows, int cols);

/* Matrix initialization on Host
Input:  number of rows and columns. Initialization seed for filling the matrix
        and reproducability. if seed = 0, matrix initialized with 0
Output: NULL in case the rows or column are not positive, or failure in memory
        allocation
        on success, a pointer to matrix type of size rows*cols with randomly
        initialized data of type data_t
*/
__host__
matrix * matrix_init(int rows, int cols, int seed);

/* Matrix initialization on Host
Input:  Matrix
Output: 0 on successfull deletion of matrix, -1 otherwise
*/
__host__
int matrix_delete(matrix *mat);

/* Accessing matrix element */
#define ELEMENT(mat, row, col) mat->data_h[(col-1)*(mat->rows) + (row-1)]

/* Matrix Multiplication on host
Input: Matrix A, B, out
Output: A*B stored in out and 0 on success. returns -1 on failure
        Assumes inner dimensions match, returns -2 if they dont match
        Assumes global memory view
*/
__host__
int MMM(matrix *A, matrix *B, matrix *out);

/*
__host__
data_t * copy_matrix(data_t *B, int Bx, int By);
*/
