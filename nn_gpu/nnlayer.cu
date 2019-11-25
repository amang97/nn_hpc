/* Copyright 2019, Aman Gupta, ENG EC 527, Prof. Martin Herbordt              */
/******************************************************************************/
/* Neural Network Layer library implementation for CUDA in C                  */
/******************************************************************************/
/* Libraries */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "matrix.cuh"
#include "nnlayer.cuh"
/******************************************************************************/
/* Parameters */
/******************************************************************************/
#define BLOCK_SIZE_W    8
#define BLOCK_SIZE_b    256
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
      for (k = 0; k < Wy; k++) {
          val += W[k*Wx+row] * dZ[k*dZx+col];
      }
      dA[row*dAx+col] = val;
    }
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
        W[row*Wx+col] -= lr*(val/Ax);
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

/* Initializing a layer with random weights and 0 bias
    Input: refrence to layer, A, W, b, Shape of W, initialization seed
    Output: W initialized randomly according to seed, bias col vector of 0 
*/
void layer_init(layer& l, int Ax, int Ay, int Wx, int Wy, int seed) {
    l.W = matrix_init(Wx, Wy, seed);
    l.b = matrix_init(Wx, 1, 0);
    l.Z = matrix_init(Ax, Wy, 0);
    l.A = matrix_allocate(Ax, Ay);
    l.dA = matrix_allocate(Ax, Ay); // H2D?
    l.dZ = matrix_allocate(Ax, Wy);
}

/* Deleting a layer
    Input: reference to layer struct
    output: 0 if freeing memory success, -1 otherwise
*/
int delete_layer(layer &l) {
    int freeW = matrix_delete(l.W);
    int freeb = matrix_delete(l.b);
    int freeZ = matrix_delete(l.Z);
    int freeA = matrix_delete(l.A);
    int freedA = matrix_delete(l.dA);
    int freedZ = matrix_delete(l.dZ);
    if ((!freeW) || (!freeb) || (!freeA) || (!freeZ) || (!freedA) || (freedZ))
        return -1;
    return 0;
}

/* Forward pass call from host */
void forward_pass_global(layer& l, data_t *A, int Ax, int Ay) {
    // copy A from Host to Device
    assert(l.A->cols == l.W->rows);
    l.A.data_h = A;
    CUDA_SAFE_CALL(cudaMemcpy(l.A.data_d, A, Ax*Ay*sizeof(data_t),
                                            cudaMemcpyHostToDevice));
    
    // call forward pass kernel
    dim3 block(BLOCK_SIZE_W, BLOCK_SIZE_W);
    dim3 grid((l.Z->rows+block.x-1)/block.x, (l.Z->cols+block.y-1)/block.y);
    FFNNFP_global<<<grid,block>>>(l.Z->data_d,
                                  l.W->data_d,
                                  l.A->data_d,
                                  l.b->data_d,
                                  l.W->rows, l.W->cols,
                                  l.A->rows, l.A->cols);
    
    // copy Z from device to host
    CUDA_SAFE_CALL(cudaMemcpy(l.Z.data_h, l.Z.data_d, Ax*Ay*sizeof(data_t),
                                  cudaMemcpyDeviceToHost));
    
}

/* backward pass call from host */
void back_propagation_global(layer& l, data_t *dZ, data_t lr) {
    int Wx = l.W->rows; int Wy = l.W->cols;
    int Ax = l.A->rows; int Ay = l.A->cols;
    int dZx = l.dZ->rows; int dZy = l.dZ->cols;
    int bx = l.b->rows;
    // copy dZ from host to device
    l.dZ.data_h = dZ;
    CUDA_SAFE_CALL(cudaMemcpy(l.dZ.data_d, dZ, dZx*dZy*sizeof(data_t),
                                            cudaMemcpyHostToDevice));
    
    // call back prop kernel calls and weight and bias update 
    dim3 block_W(BLOCK_SIZE_W, BLOCK_SIZE_W);
    dim3 grid_W((Ax+block_W.x-1)/block_W.x, (Ay+block_W.y-1)/block_W.y);
    dim3 block_b(BLOCK_SIZE_b);
    dim3 num_blocks_b((dZy*dZx+block_b.x-1)/block_b.x);
    
    FFNNBP_global<<<grid_W,block_W>>>(l.dA->data_d, l.W->data_d, l.dZ->data_d,
                                      Wx, Wy, dZx, dZy);
    // cudaDeviceSynchronize ??????????
    FFNNUb_global<<<num_blocks_b,block_b>>>(l.b->data_d, l.dZ->data_d,
                                      dZx, dZy, bx, lr);
    // cudaDeviceSynchronize ??????????
    FFNNUW_global<<<grid_W,block_W>>>(l.W->data_d, l.dZ->data_d, l.A->data_d,
                                      dZx, dZy, Ax, Ay, lr); 
    
    // copy results of dA, W, b from devic to host 
    CUDA_SAFE_CALL(cudaMemcpy(l.dA.data_h, l.dA.data_d, Ax*Ay*sizeof(data_t),
                                  cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(l.W.data_h, l.W.data_d, Wx*Wy*sizeof(data_t),
                                  cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(l.b.data_h, l.b.data_d, bx*1*sizeof(data_t),
                                  cudaMemcpyDeviceToHost));
}

/* TO DO 
Allocate memory on cuda in functions
copy values to device, send_matrix_to_device
copy values from device get_matrix_from_device
*/
