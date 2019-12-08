/* Copyright 2019, Aman Gupta, ENG EC 527, Prof. Martin Herbordt              */
/******************************************************************************/
#pragma once

#include "matrix.cuh"

typedef struct NNLayer {
    int l;                            // layer id
    Matrix *A, *W, *b, *Z, *dA, *dZ;  // Z = WA+b; dA, dZ: io error in back prop
    char f;                           // r: ReLu, s: Sigmoid
} nnlayer;

nnlayer * nnl_init(int l, int Wx, int Wy, char f);

int nnl_free(nnlayer * nnl);

__global__
void FFNNFP_global(data_t *Z, data_t *W, data_t *A, data_t *b, int Wx, int Wy,
    int Ax, int Ay);

Matrix * nnl_forward_pass_global(nnlayer * nnl, Matrix * A);

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

void relu_forward_pass_global(Matrix * A, Matrix * Z);

void sigmoid_forward_pass_global(Matrix * A, Matrix * Z);
