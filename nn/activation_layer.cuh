/* Copyright 2019, Aman Gupta, ENG EC 527, Prof. Martin Herbordt              */
/******************************************************************************/
/* Neural Network Activations library for GPU in C                            */
#pragma once    /* file guard */
/******************************************************************************/
/* Libraries */ 
#include "matrix.cuh"
#include "linear_layer.cuh"
#include <stdbool.h>
/******************************************************************************/
/* prototypes and usage */
/******************************************************************************/
typedef struct Relu_Activation {
    bool relu_activation;
    Matrix * A;
    Matrix * Z;
    Matrix * dZ;
} Relu;

typedef struct Sigmoid_Activation {
    bool sigmoid_activation;
    Matrix * A;
    Matrix * Z;
    Matrix * dZ;
} Sigmoid;

typedef struct Activation_Layer {
    Relu *r;
    Sigmoid *s;
} Activation_Layer;

/* Activations */
/******************************************************************************/
__device__
data_t relu(data_t x, data_t y);

__device__
data_t sigmoid(data_t x);

Relu * relu_activate(Linear_Layer *ll);
Sigmoid * sigmoid_activate(Linear_Layer *ll);

/* relu Activation Forward Pass*/
/******************************************************************************/
__global__
void relu_forward_global(data_t *A, data_t *Z, int Zx, int Zy);

/* Sigmoid Activation Forward Pass*/
/******************************************************************************/
__global__
void sigmoid_Forward_global(data_t *A, data_t *Z, int Zx, int Zy);

/* relu Activation Backward Pass*/
/******************************************************************************/
__global__
void relu_backward_global(data_t *dZ, data_t *dA, data_t *Z, int Zx, int Zy);

/* sigmoid Activation Backward Pass*/
/******************************************************************************/
__global__
void sigmoid_backward_global(data_t *dZ, data_t *dA, data_t *Z, int Zx, int Zy);

/* Host calls to relu */
Matrix * relu_forward_pass_global(Relu *r, Matrix * Z);

/* Host calls to sigmoid */
Matrix * relu_back_propagation_global(Relu *r, Matrix *dA, data_t lr);

/* Host calls to GPU for sigmoid for Forward pass */
Matrix * sigmoid_forward_pass_global(Sigmoid *s, Matrix *Z);

/* Host calls to GPU for sigmoid for backprop*/
Matrix * sigmoid_back_propagation_global(Sigmoid *s, Matrix *dA, data_t lr);
