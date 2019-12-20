#pragma once

#include "matrix.cuh"
#include "nn_layer.cuh"

/* Activations */
/******************************************************************************/
__device__
data_t relu(data_t x, data_t y);

__device__
data_t sigmoid(data_t x);

__global__
void relu_forward_global(data_t *A, data_t *Z, int Zx, int Zy);

__global__
void sigmoid_forward_global(data_t *A, data_t *Z, int Zx, int Zy);

__global__
void relu_backward_global(data_t *dZ, data_t *dA, data_t *Z, int Zx, int Zy);

__global__
void sigmoid_backward_global(data_t *dZ, data_t *dA, data_t *Z, int Zx, int Zy);

void relu_forward_pass_global(Matrix * A, Matrix * Z);

void sigmoid_forward_pass_global(Matrix * A, Matrix * Z);

void relu_back_propagation_global(nnlayer * nnl, Matrix *dA, data_t lr);

void sigmoid_back_propagation_global(nnlayer * nnl, Matrix *dA, data_t lr);

