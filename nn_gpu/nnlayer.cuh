/* Copyright 2019, Aman Gupta, ENG EC 527, Prof. Martin Herbordt
/******************************************************************************/
/* Neural Network Layer library for GPU in C
*/
#pragma once    /* file guard */
/******************************************************************************/
/* Libraries */
#include "matrix.cuh"
/******************************************************************************/
/* prototypes and usage */
/******************************************************************************/
typedef struct NNLayer {
    char *name;
} nnlayer;

typedef struct Layer {
    matrix *Z;  // output matrix
    matrix *W;  // weights of current layer (Weight weights)
    matrix *A;  // activation of previous layer (Activation Matrix)
    matrix *b;  // bias vector
    matrix *dA;
    matrix *dZ;
} layer;

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
    int Ax, int Ay);

/* Forward Propagation using shared memory
Input:  pointer to output data matrix Z, pointers to Weight matrix W,
        and activation matrix A, bias vector b; x,y dimensions of W and A.
        Assumes shared memory view
Output: Z = WA+b, output saved in location pointed by output data matrix
*/
__global__
void FFNNFP_shared(data_t *Z, data_t *W, data_t *A, data_t *b, int Wx, int Wy,
    int Ax, int Ay);

/* Forward Propagation using unified memory
Input:  pointer to output data matrix Z, pointers to Weight matrix W,
        and activation matrix A, bias vector b; x,y dimensions of W and A.
        Assumes unified memory view
Output: Z = WA+b, output saved in location pointed by output data matrix
*/
__global__
void FFNNFP_unified(data_t *Z, data_t *W, data_t *A, data_t *b, int Wx, int Wy,
    int Ax, int Ay);

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
    int dZx, int dZy);

/* Back Propagation using shared memory
Input:  pointer to output data matrix dA, pointers to Weight matrix W,
        and backprop out matrix dZ; x,y dimensions of W and dZ.
        Assumes shared memory view
Output: dA = W'.dZ, output saved in location pointed by output data matrix
*/
__global__
void FFNNBP_shared(data_t *dA, data_t *W, data_t *dZ, int Wx, int Wy,
    int dZx, int dZy);

/* Back Propagation using unified memory
Input:  pointer to output data matrix dA, pointers to Weight matrix W,
        and backprop out matrix dZ; x,y dimensions of W and dZ.
        Assumes unified memory view
Output: dA = W'.dZ, output saved in location pointed by output data matrix
*/
__global__
void FFNNBP_unified(data_t *dA, data_t *W, data_t *dZ, int Wx, int Wy,
    int dZx, int dZy);

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
    int Ay, data_t lr);

/* Input: pointer to updated weight matrix W, backprop out matrix dZ,
          Activation matrix A; x,y dimensions of dZ and A;
          learning rate (lr)
          Assumes shared memory view
Output: W_ = W - lr*dW; dW = (1/numData)*(dZ.A'), updates saved in W
*/
__global__
void FFNNUW_shared(data_t *W, data_t *dZ, data_t *A, int dZx, int dZy, int Ax,
    int Ay, data_t lr);

/* Input: pointer to updated weight matrix W, backprop out matrix dZ,
          Activation matrix A; x,y dimensions of dZ and A;
          learning rate (lr)
          Assumes unified memory view
Output: W_ = W - lr*dW; dW = (1/numData)*(dZ.A'), updates saved in W
*/
__global__
void FFNNUW_unified(data_t *W, data_t *dZ, data_t *A, int dZx, int dZy, int Ax,
    int Ay, data_t lr);

/* Feed Forward NN Update bias (FFNNUb) of a layer */
/******************************************************************************/
/* Input: pointer to updated bias vector b, backprop out matrix dZ;
          x,y dimensions of dZ and dim of b; learning rate (lr)
          Assumes global memory view
Output: b_ = b - lr*db; db = (1/numData)*Sum_{i=0...m}(dZi), updates saved in W
*/
__global__
void FFNNUb_global(data_t *b, data_t *dZ, int dZx, int dZy, int bx, data_t lr);

/* Input: pointer to updated bias vector b, backprop out matrix dZ;
          x,y dimensions of dZ and dim of b; learning rate (lr)
          Assumes shared memory view
Output: b_ = b - lr*db; db = (1/numData)*Sum_{i=0...m}(dZi), updates saved in W
*/
__global__
void FFNNUb_shared(data_t *b, data_t *dZ, int dZx, int dZy, int bx, data_t lr);

/* Input: pointer to updated bias vector b, backprop out matrix dZ;
          x,y dimensions of dZ and dim of b; learning rate (lr)
          Assumes unified memory view
Output: b_ = b - lr*db; db = (1/numData)*Sum_{i=0...m}(dZi), updates saved in W
*/
__global__
void FFNNUb_unified(data_t *b, data_t *dZ, int dZx, int dZy, int bx, data_t lr);

/* Initializing a layer with random weights and 0 bias
    Input: refrence to layer, A, W, b, Shape of W, initialization seed
    Output: W initialized randomly according to seed, bias col vector of 0
*/
void layer_init(layer& l, int Ax, int Ay, int Wx, int Wy, int seed);

/* Deleting a layer
    Input: Weight matrix W, bias vector b
    output: 0 if freeing memory success, -1 otherwise
*/
int delete_layer(layer& l);

/* Forward pass call from host
Input: reference to the layer
Output: pointer to matrix Z = current_layer_W' * previous_layer_A + b
*/
void forward_pas_global(layer& l, data_t *A, int Ax, int Ay);

/* backward pass call from host */
matrix * back_propagation(layer& l, data_t *dZ, data_t lr);
