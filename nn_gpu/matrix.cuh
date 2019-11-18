/* Copyright 2019, Aman Gupta, ENG EC 527, Prof. Martin Herbordt
/******************************************************************************/
/* Matrix library for GPU in C
*/
#pragma once    /* file guard */
/******************************************************************************/
/* Data Structures */
/******************************************************************************/
typedef float data_t;

typedef struct Matrix {
    int rows;
    int cols;
    data_t * data;
} matrix;

/******************************************************************************/
/* prototypes and usage */
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
matrix * matrix_init(int rows, int cols, int seed);
