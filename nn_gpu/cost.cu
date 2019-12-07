/* Copyright 2019, Aman Gupta, ENG EC 527, Prof. Martin Herbordt              */
/******************************************************************************/
/* Feed Forward Neural Network library for GPU in C                           */
/******************************************************************************/
/* Libraries */
/******************************************************************************/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "matrix.cuh"
/******************************************************************************/
/* Implementations */
/******************************************************************************/

/* Binary cross Entrpy Loss */
/******************************************************************************/
/* Loss: Binary Cross Entropy (BCE) */
/* Input: prediction array, input data point, its length (number of features)
   Output: Cost = Y*logy + (1-Y)*log(1-y)
*/
__global__
void BCELoss(data_t *cost, data_t *prediction, data_t *Y, int Ydim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Ydim) {
        // calculate partial cost (pc)
        data_t pc = (Y[i]*((data_t)log(prediction[i]))) + 
                    (((data_t)1 - Y[i])*((data_t)log((data_t)1-prediction[i])));
        atomicAdd(cost,-pc/Ydim);
    }
}

/* Loss Gradient */
/* Input: prediction, Y, and Ydim (number of features)
   Output: gradient stored in dY
*/
__global__
void dBCELoss(data_t *dY, data_t *prediction, data_t *Y, int Ydim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Ydim) {
        dY[i] = (data_t)(-1)*
        (Y[i]/prediction[i] - (((data_t)1-Y[i])/((data_t)1-prediction[i])));
    }
}