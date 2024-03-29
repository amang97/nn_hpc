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
        data_t *data_h = (data_t *)calloc(A->rows*A->cols, sizeof(data_t));
        if (!data_h) { printf("Unable to allocate matrix on host\n"); exit(1); }
        A->data_h = data_h;
        A->host_assigned = true;
    }
}

__host__
int matrix_free_host(Matrix *A) {
    if (!A) { printf("freeing NULL pointer\n"); return -1; }
    assert(A->data_h);
    free(A->data_h);
    free(A);
    return 0;
}

__host__
void matrix_allocate(Matrix *A) {
    if (!A) { printf("No matrix found to allocate on cuda\n"); exit(1); }
    matrix_allocate_host(A);
    matrix_allocate_cuda(A);
}

__host__
int matrix_free(Matrix *A) {
    int d = matrix_free_cuda(A);
    int h = matrix_free_host(A);
    if (d || h) { printf("Unable to free matrix"); return -1;}
    return 0;
}

void print_matrix(Matrix *A) {
    if (!A) { printf("No matrix found to allocate on cuda\n"); exit(1); }
    int row, col;
    for (row = 1; row <= A->rows; row++) {
        for (col = 1; col <= A->cols; col++) {
            printf("%lf,", ELEMENT(A, row, col));
        }
        printf("\n");
    }
}

void print_matrix_d(Matrix *A) {
    if (!A) { printf("No matrix found to allocate on cuda\n"); exit(1); }
    if (A->device_assigned) {
        copy_matrix_D2H(A);
        print_matrix(A);
    }
}

// /* Matrix annd cuda_utils Test */
// Matrix *A = matrix_init(3,5);
// printf("%d, %d, %s, %s, %d,  %d\n", A->rows, A->cols, A->data_h, A->data_d, A->device_assigned, A->host_assigned);
// matrix_allocate(A);
// print_matrix(A);
// printf("Test 2\n");
// Matrix *B = matrix_init(2,4);
// matrix_allocate_host(B);
// printf("Check host allocation\n");
// print_matrix(B);
// printf("check if device allocated on host allocation\n");
// print_matrix_d(B);
// printf("Allocate on device\n");
// matrix_allocate_cuda(B);
// print_matrix_d(B);
// printf("check matrix are free\n");
// int a = matrix_free(A);
// int b1 = matrix_free_host(B);
// int b2 = matrix_free_cuda(B);
// if (a || b1 || b2) printf("Unable to free matrices\n");
// printf("All matrices freed successfully\n");
