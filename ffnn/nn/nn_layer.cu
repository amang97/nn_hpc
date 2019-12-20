/* Copyright 2019, Aman Gupta, ENG EC 527, Prof. Martin Herbordt              */
/******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

#include "matrix.cuh"
#include "nn_layer.cuh"
#include "cuda_utils.cuh"
#include "../nn_param.cuh"

/******************************************************************************/
/* Neural Network layer initialization and deletion */
/******************************************************************************/
data_t rand_weight() {
    return ((data_t)rand())/((data_t)RAND_MAX);
}

void weight_init(Matrix * W) {
    int row, col;
    for (row = 1; row <= W->rows; row++) {
        for (col = 1; col <= W->cols; col++) {
            ELEMENT(W, row, col) = rand_weight();
        }
    }
    // copy host W to device
    copy_matrix_H2D(W);
}

nnlayer * nnl_init(int l, int Wx, int Wy, char f) {
    nnlayer * nnl = (nnlayer*)malloc(sizeof(nnlayer));
    if (!nnl) { printf("Unabble to initialize nn layer\n"); return NULL; }
    nnl->l = l;
    nnl->A = matrix_init(BATCH_SIZE, Wx);
    nnl->W = matrix_init(Wx, Wy);
    nnl->b = matrix_init(Wy, 1);
    nnl->Z = matrix_init(BATCH_SIZE, Wy);
    nnl->dA = matrix_init(BATCH_SIZE, Wx);
    nnl->dZ = matrix_init(BATCH_SIZE, Wy);
    matrix_allocate(nnl->A);
    matrix_allocate(nnl->W); weight_init(nnl->W);   // initialize random weights
    matrix_allocate(nnl->b);
    matrix_allocate(nnl->Z);
    matrix_allocate(nnl->dA);
    matrix_allocate(nnl->dZ);
    nnl->f = f;
    return nnl;
}

int nnl_free(nnlayer * nnl) {
    if (!nnl) { printf("Unabble to initialize nn layer\n"); return -1; }
    int freea, freew, freeb, freez, freeda, freedz;
    freea = freew = freeb = freez = freeda = freedz = -1;
    if (nnl->A) freea = matrix_free(nnl->A);
    if (nnl->W) freew = matrix_free(nnl->W);
    if (nnl->b) freeb = matrix_free(nnl->b);
    if (nnl->Z) freez = matrix_free(nnl->Z);
    if (nnl->dA) freeda = matrix_free(nnl->dA);
    if (nnl->dZ) freedz = matrix_free(nnl->dZ);
    // printf("A: %d, W: %d, b: %d, Z: %d, dA: %d, dZ: %d\n",
    //        freea, freew, freeb, freez, freeda, freedz);
    if (freea || freew || freeb || freez || freeda || freedz) return -1;
    free(nnl);
    return 0;
}

/******************************************************************************/
/* Linear Layers */
/******************************************************************************/
/* Forward Pass */
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

/* Forward pass call from host */
Matrix * nnl_forward_pass_global(nnlayer * nnl, Matrix *A) {
    assert(nnl->W->rows == A->cols); nnl->A = A;

    // call forward pass kernel
    dim3 block_W(BLOCK_SIZE_W, BLOCK_SIZE_W);
    dim3 grid_W((nnl->Z->rows+block_W.x-1)/block_W.x,
                (nnl->Z->cols+block_W.y-1)/block_W.y);
    FFNNFP_global<<<grid_W, block_W>>>(nnl->Z->data_d,
                                        nnl->W->data_d,
                                        nnl->A->data_d,
                                        nnl->b->data_d,
                                        nnl->W->rows, nnl->W->cols,
                                        nnl->A->rows, nnl->A->cols);

    return nnl->Z;
}

/* Back Propagation */
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
        W[row*Wx+col] += lr*(val/Ax);
    }
}

__global__
void FFNNUb_global(data_t *b, data_t *dZ, int dZx, int dZy, int bx, data_t lr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dZx*dZy) {
        int zx = i % dZx;
        int zy = i / dZx;
        // do an atomic add to avoid race conditions
        // (because many threads might write to same memory location  )
        atomicAdd(&b[zy], lr*(dZ[zy*dZx+zx]/dZx));
    }

}

/* backward pass call from host */
void nnl_back_propagation_global(nnlayer * nnl, Matrix *dZ, data_t lr) {
    // to reduce number of memory references
    int Ax = nnl->A->rows, Ay = nnl->A->cols;
    int Wx = nnl->W->rows, Wy = nnl->W->cols;
    int dZx = dZ->rows, dZy = dZ->cols;

    // call forward pass kernel
    // Compute back-propagation error using dZ
    dim3 block_W(BLOCK_SIZE_W, BLOCK_SIZE_W);
    dim3 grid_W((Ax+block_W.x-1)/block_W.x, (Ay+block_W.y-1)/block_W.y);
    FFNNBP_global<<<grid_W, block_W>>>(nnl->dA->data_d,
                                        nnl->W->data_d,
                                        dZ->data_d,
                                        Wx, Wy,
                                        dZx, dZy);
    
    // update bias
    dim3 block_b(BLOCK_SIZE_b);
    dim3 num_blocks_b((dZy*dZx+block_b.x-1)/block_b.x);
    FFNNUb_global<<<num_blocks_b, block_b>>>(nnl->b->data_d,
                                            dZ->data_d,
                                            dZx, dZy,
                                            nnl->b->rows,
                                            lr);
    
    // update Weights
    FFNNUW_global<<<grid_W, block_W>>>(nnl->W->data_d,
                                        dZ->data_d,
                                        nnl->A->data_d,
                                        dZx, dZy,
                                        Ax, Ay,
                                        lr);

    // return nnl->dA;
}


// /* Testing network and layer initializations */
// printf("On host\n");
// printf("Stats on host for layer %d, Activation: %c\n", nn->layer[0]->l, nn->layer[0]->f);
// print_matrix(nn->layer[0]->A);
// printf("\n");
// print_matrix(nn->layer[0]->W);
// printf("\n");
// print_matrix(nn->layer[0]->b);
// printf("\n");
// print_matrix(nn->layer[0]->Z);
// printf("\n");
// print_matrix(nn->layer[0]->dA);
// printf("\n");
// print_matrix(nn->layer[0]->dZ);
// printf("\n\n");
// printf("Stats on host for layer %d, Activation: %c\n", nn->layer[1]->l, nn->layer[1]->f);
// print_matrix(nn->layer[1]->A);
// printf("\n");
// print_matrix(nn->layer[1]->W);
// printf("\n");
// print_matrix(nn->layer[1]->b);
// printf("\n");
// print_matrix(nn->layer[1]->Z);
// printf("\n");
// print_matrix(nn->layer[1]->dA);
// printf("\n");
// print_matrix(nn->layer[1]->dZ);
// printf("\n\n");

// printf("On Device\n");
// printf("Stats on device for layer %d, Activation: %c\n", nn->layer[0]->l, nn->layer[0]->f);
// print_matrix_d(nn->layer[0]->A);
// printf("\n");
// print_matrix_d(nn->layer[0]->W);
// printf("\n");
// print_matrix_d(nn->layer[0]->b);
// printf("\n");
// print_matrix_d(nn->layer[0]->Z);
// printf("\n");
// print_matrix_d(nn->layer[0]->dA);
// printf("\n");
// print_matrix_d(nn->layer[0]->dZ);
// printf("\n\n");
// printf("Stats on device for layer %d, Activation: %c\n", nn->layer[1]->l, nn->layer[1]->f);
// print_matrix_d(nn->layer[1]->A);
// printf("\n");
// print_matrix_d(nn->layer[1]->W);
// printf("\n");
// print_matrix_d(nn->layer[1]->b);
// printf("\n");
// print_matrix_d(nn->layer[1]->Z);
// printf("\n");
// print_matrix_d(nn->layer[1]->dA);
// printf("\n");
// print_matrix_d(nn->layer[1]->dZ);
// printf("\n\n");