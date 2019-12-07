/* Copyright 2019, Aman Gupta, ENG EC 527, Prof. Martin Herbordt              */
/******************************************************************************/
/* Neural Network Activations library for GPU in C                            */
/******************************************************************************/
/* Libraries */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "matrix.cuh"
#include "nn_param.cuh"
#include "linear_layer.cuh"
#include "activation_layer.cuh"
#include "cuda_utils.cuh"
/******************************************************************************/
/* Implementations */
/******************************************************************************/

/* Activations */
/******************************************************************************/
__device__
data_t relu(data_t x, data_t y) {
    return (x > y) ? x : y;
}

__device__
data_t sigmoid(data_t x) {
    return ((data_t)1) / ((data_t)1 + (data_t)exp(-x));
}

/* RELU Activation Forward pass */
/******************************************************************************/
__global__
void relu_forward_global(data_t *A, data_t *Z, int Zx, int Zy) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < Zx * Zy) {
        A[index] = relu(Z[index],(data_t)0);
    }
}

/* Sigmoid Activation Forward Pass*/
/******************************************************************************/
__global__
void sigmoid_forward_global(data_t *A, data_t *Z, int Zx, int Zy) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < Zx * Zy) {
        A[index] = sigmoid(Z[index]);
    }
}

/* RELU Activation Backward Pass*/
/******************************************************************************/
__global__
void relu_backward_global(data_t *dZ, data_t *dA, data_t *Z, int Zx, int Zy) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < Zx * Zy) {
        dZ[index] = (Z[index] > 0) ? dA[index] : 0;
    }
}

/* Sigmoid Activation Backward Pass*/
/******************************************************************************/
__global__
void sigmoid_backward_global(data_t *dZ, data_t *dA, data_t *Z, int Zx, int Zy) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < Zx*Zy) {
        dZ[index] = dA[index]*sigmoid(Z[index])*((data_t)1 - sigmoid(Z[index]));
    }
}

Relu * relu_activate(Linear_Layer *ll) {
    Relu * r = (Relu *)malloc(sizeof(Relu));
    if (!r) return NULL;
    r->A = matrix_init(ll->A->rows, ll->A->cols);
    int Zx = ll->Z->rows, Zy = ll->Z->cols;
    r->Z = matrix_init(Zx, Zy);
    r->dZ = matrix_init(Zx, Zy);
    r->relu_activation = true;
    return r;
}

Sigmoid * sigmoid_activate(Linear_Layer *ll) {
    Sigmoid * s = (Sigmoid *)malloc(sizeof(Sigmoid));
    if (!s) return NULL;
    s->A = matrix_init(ll->A->rows, ll->A->cols);
    int Zx = ll->Z->rows, Zy = ll->Z->cols;
    s->Z = matrix_init(Zx, Zy);
    s->dZ = matrix_init(Zx, Zy);
    s->sigmoid_activation = true;
    return s;
}

/* Host calls to GPU for RELU for forward pass*/
Matrix * relu_forward_pass_global(Relu *r, Matrix * Z) {
    int  Zx = Z->rows, Zy = Z->cols;
    r->Z = Z;
    // printf("\n\ninside relu forward pass\n");
    // print_matrix(r->Z);
    // printf("\n\n");
    matrix_allocate(r->A, Zx, Zy);
    dim3 block(BLOCK_SIZE_b);
    dim3 num_blocks((Zy*Zx+block.x-1)/block.x);
    relu_forward_global<<<num_blocks,block>>>(r->A->data_d,
                                            r->Z->data_d,
                                            Zx, Zy);
    // copy_matrix_D2H(r->A);
    // printf("\n\nRelu Output\n");
    // print_matrix(r->A);
    return r->A;
}

/* Host calls to GPU for RELU for backProp */
Matrix * relu_back_propagation_global(Relu *r, Matrix *dA, data_t lr) {
    int Zx = r->Z->rows; int Zy = r->Z->cols;
    matrix_allocate(r->dZ, Zx, Zy);
    dim3 block(BLOCK_SIZE_b);
    dim3 num_blocks((Zy*Zx+block.x-1)/block.x);
    relu_backward_global<<<num_blocks,block>>>(r->dZ->data_d,
                                                dA->data_d,
                                                r->Z->data_d,
                                                Zx, Zy);
    return r->dZ;
}

/* Host calls to GPU for Sigmoid for Forward pass */
Matrix * sigmoid_forward_pass_global(Sigmoid *s, Matrix *Z) {
    int Zx = Z->rows; int Zy = Z->cols;
    s->Z = Z;
    // printf("\n\ninside sigmoid forward pass\n");
    // print_matrix(s->Z);
    // printf("\n\n");
    matrix_allocate(s->A, Zx, Zy);
    dim3 block(BLOCK_SIZE_b);
    dim3 num_blocks((Zy*Zx+block.x-1)/block.x);
    sigmoid_forward_global<<<num_blocks,block>>>(s->A->data_d,
                                                s->Z->data_d,
                                                Zx, Zy);
    // copy_matrix_D2H(s->A);
    // printf("\n\nSigmoid Output\n");
    // print_matrix(s->A);
    return s->A;
}

/* Host calls to GPU for Sigmoid for backprop*/
Matrix * sigmoid_back_propagation_global(Sigmoid *s, Matrix *dA, data_t lr) {
    int Zx = s->Z->rows; int Zy = s->Z->cols;
    matrix_allocate(s->dZ, Zx, Zy);
    dim3 block(BLOCK_SIZE_b);
    dim3 num_blocks((Zy*Zx+block.x-1)/block.x);
    sigmoid_backward_global<<<num_blocks,block>>>(s->dZ->data_d,
                                                dA->data_d,
                                                s->Z->data_d,
                                                Zx, Zy);
    return s->dZ;
}
