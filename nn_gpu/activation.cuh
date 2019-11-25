/* Copyright 2019, Aman Gupta, ENG EC 527, Prof. Martin Herbordt              */
/******************************************************************************/
/* Neural Network Activations library for GPU in C                            */
#pragma once    /* file guard */
/******************************************************************************/
/* Libraries */
/******************************************************************************/
/* prototypes and usage */
/******************************************************************************/

/* Activations */
/******************************************************************************/
__device__
data_t relu(data_t x, data_t y);

__device__
data_t sigmoid(data_t x);


/* RELU Activation Forward Pass*/
/******************************************************************************/
__global__
void RELU_forward_global(data_t *A, data_t *Z, int Zx, int Zy);

__global__
void RELU_forward_shared(data_t *A, data_t *Z, int Zx, int Zy);

__global__
void RELU_forward_unified(data_t *A, data_t *Z, int Zx, int Zy);

/* Sigmoid Activation Forward Pass*/
/******************************************************************************/
__global__
void Sigmoid_Forward_global(data_t *A, data_t *Z, int Zx, int Zy);

__global__
void Sigmoid_Forward_shared(data_t *A, data_t *Z, int Zx, int Zy);

__global__
void Sigmoid_Forward_unified(data_t *A, data_t *Z, int Zx, int Zy);

/* RELU Activation Backward Pass*/
/******************************************************************************/
__global__
void RELU_backward_global(data_t *dZ, data_t *dA, data_t *Z, int Zx, int Zy);

__global__
void RELU_backward_shared(data_t *dZ, data_t *dA, data_t *Z, int Zx, int Zy);

__global__
void RELU_backward_unified(data_t *dZ, data_t *dA, data_t *Z, int Zx, int Zy);

/* Sigmoid Activation Backward Pass*/
/******************************************************************************/
__global__
void Sigmoid_backward_global(data_t *dZ, data_t *dA, data_t *Z, int Zx, int Zy);

__global__
void Sigmoid_backward_shared(data_t *dZ, data_t *dA, data_t *Z, int Zx, int Zy);

__global__
void Sigmoid_backward_unified(data_t *dZ, data_t *dA, data_t *Z, int Zx, int Zy);

