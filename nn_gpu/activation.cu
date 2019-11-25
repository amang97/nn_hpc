/* Copyright 2019, Aman Gupta, ENG EC 527, Prof. Martin Herbordt              */
/******************************************************************************/
/* Neural Network Activations library for GPU in C                            */
/******************************************************************************/
/* Libraries */
#include <math.h>
#include "matrix.cuh"
/******************************************************************************/
/* Parameters */
/******************************************************************************/
#define BLOCK_SIZE_b    256
/******************************************************************************/
/* Implementations */
/******************************************************************************/

/* Activations */
/******************************************************************************/
__device__
data_t relu(data_t x, data_t y) {
    return (x <= y) ? x : y;
}

__device__
data_t sigmoid(data_t x) {
    return ((data_t)1) / ((data_t)1 + (data_t)exp(-x));
}

/* RELU Activation Forward pass */
/******************************************************************************/
__global__
void RELU_forward_global(data_t *A, data_t *Z, int Zx, int Zy) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < Zx * Zy) {
        A[index] = relu(Z[index],(data_t)0);
    }
}

__global__
void RELU_forward_shared(data_t *A, data_t *Z, int Zx, int Zy) {

}

__global__
void RELU_forward_unified(data_t *A, data_t *Z, int Zx, int Zy) {

}

/* Sigmoid Activation Forward Pass*/
/******************************************************************************/
__global__
void Sigmoid_Forward_global(data_t *A, data_t *Z, int Zx, int Zy) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < Zx * Zy) {
        A[index] = sigmoid(Z[index]);
    }
}

__global__
void Sigmoid_Forward_shared(data_t *A, data_t *Z, int Zx, int Zy) {

}

__global__
void Sigmoid_Forward_unified(data_t *A, data_t *Z, int Zx, int Zy) {

}

/* RELU Activation Backward Pass*/
/******************************************************************************/
__global__
void RELU_backward_global(data_t *dZ, data_t *dA, data_t *Z, int Zx, int Zy) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < Zx * Zy) {
        dZ[index] = (Z[index] > 0) ? dA[index] : 0;
    }
}

__global__
void RELU_backward_shared(data_t *dZ, data_t *dA, data_t *Z, int Zx, int Zy);

__global__
void RELU_backward_unified(data_t *dZ, data_t *dA, data_t *Z, int Zx, int Zy);

/* Sigmoid Activation Backward Pass*/
/******************************************************************************/
__global__
void Sigmoid_backward_global(data_t *dZ, data_t *dA, data_t *Z, int Zx, int Zy) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < Zx*Zy) {
        dZ[index] = dA[index]*sigmoid(Z[index])*((data_t)1 - sigmoid(Z[index]));
    }
}

__global__
void Sigmoid_backward_shared(data_t *dA, data_t *dZ, data_t *Z, int Zx, int Zy);

__global__
void Sigmoid_backward_unified(data_t *dA, data_t *dZ, data_t *Z, int Zx, int Zy);

/* Host calls to GPU for RELU for forward pass*/
matrix * RELU_forward(layer& l) {
    data_t *Z = l.Z->data; int Zx = l.Z->rows; int Zy = l.Z->cols;
    dim3 block(BLOCK_SIZE_b);
    dim3 num_blocks((Zy*Zx+block.x-1)/block.x);
    RELU_forward_global<<<num_blocks,block>>>(l.A->data, Z, Zx, Zy);
    
    return (matrix *)l.A;
}

/* Host calls to GPU for RELU for backProp */
matrix * RELU_back_propagation(layer& l, data_t lr) {
    data_t *Z = l.Z->data; int Zx = l.Z->rows; int Zy = l.Z->cols;
    dim3 block(BLOCK_SIZE_b);
    dim3 num_blocks((Zy*Zx+block.x-1)/block.x);
    RELU_backward_global<<<num_blocks,block>>>(l.dZ->data,l.dA->data,Z,Zx,Zy);

    return (matrix *)l.dZ;
}

/* Host calls to GPU for Sigmoid for Forward pass */
matrix * Sigmoid_forward(layer& l) {
    data_t *Z = l.Z->data; int Zx = l.Z->rows; int Zy = l.Z->cols;
    dim3 block(BLOCK_SIZE_b);
    dim3 num_blocks((Zy*Zx+block.x-1)/block.x);
    Sigmoid_Forward_global<<<num_blocks,block>>>(l.A->data, Z, Zx, Zy);
    
    return (matrix *)l.A;
}

/* Host calls to GPU for Sigmoid for backprop*/
matrix * Sigmoid_back_propagation(layer& l, data_t lr) {
    data_t *Z = l.Z->data; int Zx = l.Z->rows; int Zy = l.Z->cols;
    dim3 block(BLOCK_SIZE_b);
    dim3 num_blocks((Zy*Zx+block.x-1)/block.x);
    Sigmoid_backward_global<<<num_blocks,block>>>(l.dZ->data,l.dA->data,Z,Zx,Zy);

    return (matrix *)l.dZ;
}