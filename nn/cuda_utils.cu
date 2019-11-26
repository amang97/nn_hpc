#include "cuda_utils.cuh"
#include "nn_param.cuh"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

void matrix_allocate_cuda(Matrix *A) {
    if (!A->device_assigned) {
        data_t *data_d = NULL;
        int N = A->rows*A->cols;
        CUDA_SAFE_CALL(cudaMalloc((void **)&data_d, N*sizeof(data_t)));
        A->data_d = data_d;
        A->device_assigned = true;
    }
}

int matrix_free_cuda(Matrix *A) {
    if (!A) {printf("freeing NULL pointer\n"); return -1;
    assert(A->data_d);
    CUDA_SAFE_CALL(cudaFree(A->data_d));
    return 0;
}

void copy_matrix_H2D(Matrix *A) {
    if (A->device_assigned && A->host_assigned) {
        int N = A->rows*A->cols;
        CUDA_SAFE_CALL(cudaMemcpy(A->data_d, A->data_h, N*sizeof(data_t),
        cudaMemcpyHostToDevice));
    } else {printf("unable to copy data from host to device");}
}

void copy_matrix_D2H(Matrix *A) {
    if (A->device_assigned && A->host_assigned) {
        int N = A->rows*A->cols;
        CUDA_SAFE_CALL(cudaMemcpy(A->data_h, A->data_d, N*sizeof(data_t),
        cudaMemcpyDeviceToHost));
    } else {printf("unable to copy data from host to device");}
}