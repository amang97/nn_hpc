/* Copyright 2019, Aman Gupta, ENG EC 527, Prof. Martin Herbordt              */
/******************************************************************************/
/* Matrix library for GPU in C */
#pragma once    /* file guard */

#include "../nn_param.cuh"
#include <stdbool.h>

/* Accessing matrix element from device data */
#define ELEMENT(mat, row, col) mat->data_h[(row-1)*(mat->cols) + (col-1)]

/******************************************************************************/
/* Data Structures */
/******************************************************************************/
typedef struct Matrix {
    int rows, cols;
    data_t * data_d, * data_h;
    bool device_assigned, host_assigned;
} Matrix;

/******************************************************************************/
/* prototypes and usage */
/******************************************************************************/
__host__
Matrix * matrix_init(int rows, int cols);

__host__
void matrix_allocate_host(Matrix *A);

__host__
int matrix_free_host(Matrix *A);

__host__
void matrix_allocate(Matrix *A);

__host__
int matrix_free(Matrix *A);

void print_matrix(Matrix *A);

void print_matrix_d(Matrix *A);