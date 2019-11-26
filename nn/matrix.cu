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
#include "cuda_utils.cuh"
/******************************************************************************/
/* Implementations */
/******************************************************************************/

__host__
Matrix * matrix_init(int rows, int cols) {
    // small check to ensure rows and cols are positive numbers
    if (rows <= 0 || cols <= 0) return NULL;

    Matrix *m = (Matrix *)malloc(sizeof(Matrix));
    m -> rows = rows;
    m -> cols = cols;
    m -> data_d = NULL;
    m -> data_h = NULL;
    m -> device_assigned = false;
    m -> host_assigned = false;
    return m;
}

__host__
void matrix_allocate_host(Matrix *A) {
    if (!A->host_assigned) {
        data_t *data_h = (data_t *)calloc(A->rows*A->cols,sizeof(data_t));
        if (!data_h) {printf("Unable to allocate matrix\n"); exit(-1);}
        A->data_h = data_h;
        A->host_assigned = true;
    }
}

__host__
int matrix_free_host(Matrix *A) {
    if (!A) {printf("freeing NULL pointer\n"); return -1;}
    assert(A->data_h);
    free(A->data_h);
    free(A);
    return 0;
}

__host__
void matrix_allocate(Matrix *A, int rows, int cols) {
    if (!A->device_assigned && !A->host_assigned) {
        A->rows = rows;
        A->cols = cols;
        matrix_allocate_cuda(A);
        matrix_allocate_host(A);   
    }
}

__host__
int matrix_free(Matrix *A) {
    int d = matrix_free_cuda(A);
    int h = matrix_free_host(A);
    if (d || h) {printf("Unable to free matrix"); return -1;}
    return 0;
}

__host__
void print_matrix(Matrix *A) {
    int row, col;
    for (row = 0; row < A->rows; row++) {
        for (col = 0; col < A->cols; col++) {
            printf("%lf",ELEMENT(A,row,col));
        }
        printf("\n");
    }
}
