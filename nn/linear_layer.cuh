/* Copyright 2019, Aman Gupta, ENG EC 527, Prof. Martin Herbordt
/******************************************************************************/
/* Neural Network Layer library for GPU in C
*/
#pragma once    /* file guard */
/******************************************************************************/
/* Libraries */
#include "matrix.cuh"
#include "nn_param.cuh"
/******************************************************************************/
/* prototypes and usage */
/******************************************************************************/
/* Linear Layer (ll) */
typedef struct LinearLayer {
    int seed;
    Matrix *Z;  // output matrix
    Matrix *W;  // weights of current layer (Weight weights)
    Matrix *A;  // activation of previous layer (Activation Matrix)
    Matrix *b;  // bias vector
    Matrix *dA;
} Linear_Layer;

data_t rand_weight();

void weight_init(Linear_Layer *ll);

void bias_init(Linear_Layer *ll);

void ll_init(Linear_Layer *ll, int Ax, int Ay, int Wx, int Wy, int seed);

int ll_free(Linear_Layer *ll);

__global__
void FFNNFP_global(data_t *Z, data_t *W, data_t *A, data_t *b, int Wx, int Wy,
    int Ax, int Ay);

__global__
void FFNNBP_global(data_t *dA, data_t *W, data_t *dZ, int Wx, int Wy,
    int dZx, int dZy);

__global__
void FFNNUW_global(data_t *W, data_t *dZ, data_t *A, int dZx, int dZy, int Ax,
    int Ay, data_t lr);

__global__
void FFNNUb_global(data_t *b, data_t *dZ, int dZx, int dZy, int bx, data_t lr);

Matrix * ll_forward_pass_global(Linear_Layer * ll, Matrix *A);

Matrix * ll_back_propagation_global(Linear_Layer * ll, Matrix *dZ, data_t lr);

Matrix getW(Linear_Layer *ll);

Matrix getb(Linear_Layer *ll);

int getWx(Linear_Layer *ll);

int getWy(Linear_Layer *ll);
