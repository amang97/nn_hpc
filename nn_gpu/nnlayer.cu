/* Copyright 2019, Aman Gupta, ENG EC 527, Prof. Martin Herbordt              */
/******************************************************************************/
/* Neural Network Layer library implementation for CUDA in C                  */
/******************************************************************************/
/* Libraries */
#include <stdio.h>
#include <stdlib.h>
#include "matrix.cuh"
#include "layer.cuh"

/******************************************************************************/
/* Parameters */
#define TILE_WIDTH              32
#define NUM_THREADS_PER_BLOCK   1024

/******************************************************************************/
/* Implementations */
/******************************************************************************/

/* Feed Forward NN Forward pass (FFNNFP) on a layer */
/******************************************************************************/
/* Forward Propagation using global memory
Input:  pointer to output data matrix Z, pointers to Weight matrix W,
        and activation matrix A, bias vector b; x,y dimensions of W and A.
        Assumes global memory view
Output: Z = WA+b, output saved in location pointed by output data matrix
*/
__global__
void FFNNFP_global(data_t *Z, data_t *W, data_t *A, data_t *b, int Wx, int Wy,
    int Ax, int Ay) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockIdx.x + threadIdx.x;
    int Zx = Ax;
    int Zy = Wy;
    data_t val = (data_t)0;
    
    int k;
    if (row < Zy && col < Zx) {
        for (k = 0; k < Wx; k++) {
            val += W[row*Wx+k] * A[k*Ax+col];
        }
        Z[row*Zx+col] = val + b[row];
    }
}

/* Forward Propagation using shared memory
Input:  pointer to output data matrix Z, pointers to Weight matrix W,
        and activation matrix A, bias vector b; x,y dimensions of W and A.
        Assumes shared memory view
Output: Z = WA+b, output saved in location pointed by output data matrix
*/
__global__
void FFNNFP_shared(data_t *Z, data_t *W, data_t *A, data_t *b, int Wx, int Wy,
    int Ax, int Ay) {

}

/* Forward Propagation using unified memory
Input:  pointer to output data matrix Z, pointers to Weight matrix W,
        and activation matrix A, bias vector b; x,y dimensions of W and A.
        Assumes unified memory view
Output: Z = WA+b, output saved in location pointed by output data matrix
*/
__global__
void FFNNFP_unified(data_t *Z, data_t *W, data_t *A, data_t *b, int Wx, int Wy,
    int Ax, int Ay) {

}

/* Feed Forward NN Back Prop (FFNNBP) on a layer */
/******************************************************************************/
/* Back Propagation using global memory
Input:  pointer to output data matrix dA, pointers to Weight matrix W,
        and backprop out matrix dZ; x,y dimensions of W and dZ.
        Assumes global memory view
Output: dA = W'.dZ, output saved in location pointed by output data matrix
*/
__global__
void FFNNBP_global(data_t *dA, data_t *W, data_t *dZ, int Wx, int Wy,
    int dZx, int dZy) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // indexing in W transposed
    int dAx = dZx;
    int dAy = Wx;
    data_t val = (data_t)0;

    int k;
    if (row < dAy && col < dAx) {
        val += W[k*Wx+row] * dZ[i*dZx+col];
    }
    dA[row*dAx+col] = val;
}

/* Back Propagation using shared memory
Input:  pointer to output data matrix dA, pointers to Weight matrix W,
        and backprop out matrix dZ; x,y dimensions of W and dZ.
        Assumes shared memory view
Output: dA = W'.dZ, output saved in location pointed by output data matrix
*/
__global__
void FFNNBP_shared(data_t *dA, data_t *W, data_t *dZ, int Wx, int Wy,
    int dZx, int dZy) {

}

/* Back Propagation using unified memory
Input:  pointer to output data matrix dA, pointers to Weight matrix W,
        and backprop out matrix dZ; x,y dimensions of W and dZ.
        Assumes unified memory view
Output: dA = W'.dZ, output saved in location pointed by output data matrix
*/
__global__
void FFNNBP_unified(data_t *dA, data_t *W, data_t *dZ, int Wx, int Wy,
    int dZx, int dZy) {
    
}

/******************************************************************************/
/*************************** GRADIENT DESCENT *********************************/
/******************************************************************************/
/* Feed Forward NN Update weights (FFNNUW) of a layer */
/******************************************************************************/
/* Input: pointer to updated weight matrix W, backprop out matrix dZ,
          Activation matrix A; x,y dimensions of dZ and A;
          learning rate (lr)
          Assumes global memory view
Output: W_ = W - lr*dW; dW = (1/numData)*(dZ.A'), updates saved in W
*/
__global__
void FFNNUW_global(data_t *W, data_t *dZ, data_t *A, int dZx, int dZy, int Ax,
    int Ay, data_t lr) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // indexing in A transposed
    int Wx = Ay;
    int Wy = dZy;
    data_t val = (data_t)0;

    int k;
    if (row < Wy && col < Wx) {
        for (k = 0; k < dZx; k++) {
            val += dZ[row*dZx+k] * A[col*Ax+k];
        }
        W[row*Wx+col] -= -lr*(val/Ax);
    }
}

/* Input: pointer to updated weight matrix W, backprop out matrix dZ,
          Activation matrix A; x,y dimensions of dZ and A;
          learning rate (lr)
          Assumes shared memory view
Output: W_ = W - lr*dW; dW = (1/numData)*(dZ.A'), updates saved in W
*/
__global__
void FFNNUW_shared(data_t *W, data_t *dZ, data_t *A, int dZx, int dZy, int Ax,
    int Ay, data_t lr) {

}

/* Input: pointer to updated weight matrix W, backprop out matrix dZ,
          Activation matrix A; x,y dimensions of dZ and A;
          learning rate (lr)
          Assumes unified memory view
Output: W_ = W - lr*dW; dW = (1/numData)*(dZ.A'), updates saved in W
*/
__global__
void FFNNUW_unified(data_t *W, data_t *dZ, data_t *A, int dZx, int dZy, int Ax,
    int Ay, data_t lr) {

}

/* Feed Forward NN Update bias (FFNNUb) of a layer */
/******************************************************************************/
/* Input: pointer to updated bias vector b, backprop out matrix dZ;
          x,y dimensions of dZ and dim of b; learning rate (lr)
          Assumes global memory view
Output: b_ = b - lr*db; db = (1/numData)*Sum_{i=0...m}(dZi), updates saved in W
*/
__global__
void FFNNUb_global(data_t *b, data_t *dZ, int dZx, int dZy, int bx, data_t lr) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dZx*dZy) {
        int zx = index % dZx;
        int zy = index / dZx;
        // do an atomic add to avoid race conditions 
        // (because many threads might write to same memory location  )
        atomicAdd(&b[zy],-lr*(dZ[zy*dZx+zx]/dZx));
    }

}

/* Input: pointer to updated bias vector b, backprop out matrix dZ;
          x,y dimensions of dZ and dim of b; learning rate (lr)
          Assumes shared memory view
Output: b_ = b - lr*db; db = (1/numData)*Sum_{i=0...m}(dZi), updates saved in W
*/
__global__
void FFNNUb_shared(data_t *b, data_t *dZ, int dZx, int dZy, int bx, data_t lr) {

}

/* Input: pointer to updated bias vector b, backprop out matrix dZ;
          x,y dimensions of dZ and dim of b; learning rate (lr)
          Assumes unified memory view
Output: b_ = b - lr*db; db = (1/numData)*Sum_{i=0...m}(dZi), updates saved in W
*/
__global__
void FFNNUb_unified(data_t *b, data_t *dZ, int dZx, int dZy, int bx, data_t lr) {

}
