/* Copyright 2019, Aman Gupta, ENG EC 527, Prof. Martin Herbordt
/******************************************************************************/
/* Feed Forward Neural Network library for GPU in C                           */
#pragma once    /* file guard */
/******************************************************************************/
/* Libraries */
#include "matrix.cuh"
#include "nnlayer.cuh"
#include "activation.cuh"
/******************************************************************************/
/* prototypes and usage */
/******************************************************************************/
typedef struct NNLoss {
    data_t *(loss)(data_t *cost, data_t *prediction, data_t *x, int Xdim);
    matrix *(dloss)(data_t *dY, data_t *prediction, data_t *x, int Xdim);
} nnloss;

/* Binary cross Entrpy Loss */
/******************************************************************************/
/* Loss: Binary Cross Entropy (BCE) */
/* Input: prediction array, input data point, its length (number of features)
   Output: Cost    
*/
__global__
void BCELoss(data_t *cost, data_t *prediction, data_t *x, int Xdim);

/* Loss Gradient */
/* Input: prediction, x, and Xdim (number of features)
   Output: gradient stored in dY
*/
__global__
void dBCELoss(data_t *dY, data_t *prediction, data_t *x, int Xdim);

nnloss bce;
bce.loss = BCELoss;
bce.dloss = dBCELoss;

typedef struct FFNN {
    nnlayer *layers;
    nnloss *bce;
    data_t learning_rate;
} ffnn;



