/* Copyright 2019, Aman Gupta, ENG EC 527, Prof. Martin Herbordt              */
/******************************************************************************/
/* Neural Network Layer library implementation for CUDA in C                  */
/******************************************************************************/
/* Libraries */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include "matrix.cuh"
#include "cuda_utils.cuh"
#include "linear_layer.cuh"
#include "cuPrintf.cuh"
/******************************************************************************/
/* Implementations */
/******************************************************************************/

data_t rand_weight() {
    return ((data_t)rand())/((data_t)RAND_MAX);
}

void weight_init(Linear_Layer *ll) {
    int row, col;
    for (row = 1; row <= ll->W->rows; row++) {
        for (col = 1; col <= ll->W->cols; col++) {
            ELEMENT_H(ll->W, row, col) = rand_weight();
        }
    }
    // copy host W to device
    copy_matrix_H2D(ll->W);
}

void bias_init(Linear_Layer *ll) {
    int row;
    for (row = 1; row <= ll->b->rows; row++) {
        ELEMENT_H(ll->b, row, 1) = (data_t)0;
    }
    // copy host b to device
    copy_matrix_H2D(ll->b);
}

void ll_init(Linear_Layer *ll, int Ax, int Ay, int Wx, int Wy) {
    // initialize W and b
    ll->Z = matrix_init(Ax, Wy);
    ll->W = matrix_init(Wx, Wy);
    ll->A = matrix_init(Ax, Ay);
    ll->b = matrix_init(Wy, 1);
    ll->dA = matrix_init(Ax, Ay);

    // Allocate W and b on host and device
    matrix_allocate_cuda(ll->W);
    matrix_allocate_host(ll->W);
    matrix_allocate_cuda(ll->b);
    matrix_allocate_host(ll->b);

    // initialize W and b data and copy data H2D
    weight_init(ll); // initialized with random values
    bias_init(ll);  // initialized with 0
}

int ll_free(Linear_Layer *ll) {
    if (!ll) { printf("Couldnt find the layer pointer\n"); return -1; }
    int freez, freew, freea, freeb, freeda;
    freez = freew = freea = freeb = freeda = -1;
    if (ll->Z) freez = matrix_free(ll->Z);
    if (ll->W) freew = matrix_free(ll->W);
    if (ll->A) freea = matrix_free(ll->A);
    if (ll->b) freeb = matrix_free(ll->b);
    if (ll->dA) freeda = matrix_free(ll->dA);
    printf("%d, %d, %d, %d, %d", freez, freew, freea, freeb, freeda);
    if (freez || freew || freea || freeb || freeda) return -1;
    free(ll);
    return 0;
}

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
        //cuPrintf("%d\n", val + b[row]);
    }
}

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
        W[row*Wx+col] -= lr*(val/Ax);
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
        atomicAdd(&b[zy],-lr*(dZ[zy*dZx+zx]/dZx));
    }

}

/* Forward pass call from host */
Matrix * ll_forward_pass_global(Linear_Layer * ll, Matrix *A) {
    // copy A from Host to Device
    assert(ll->W->rows == A->cols); ll->A = A;

    // Allocate Z if not allocated yet
    matrix_allocate(ll->Z, A->rows, ll->W->cols);
    // copy_matrix_D2H(ll->A);
    // printf("\n\nLinear Layer Input\n");
    // print_matrix(ll->A);
    // printf("\n\n");
    // call forward pass kernel
    dim3 block_W(BLOCK_SIZE_W, BLOCK_SIZE_W);
    dim3 grid_W((ll->Z->rows+block_W.x-1)/block_W.x,
                (ll->Z->cols+block_W.y-1)/block_W.y);
    FFNNFP_global<<<grid_W, block_W>>>(ll->Z->data_d,
                                        ll->W->data_d,
                                        A->data_d,
                                        ll->b->data_d,
                                        ll->W->rows, ll->W->cols,
                                        ll->A->rows, ll->A->cols);
    copy_matrix_D2H(ll->Z);
    //printf("ZSize: %d\n", ll->Z->rows*ll->Z->cols);
    printf("\n\nLinear Layer Output\n");
    print_matrix(ll->Z);
    printf("\n\n");
    return ll->Z;
}

/* backward pass call from host */
Matrix * ll_back_propagation_global(Linear_Layer * ll, Matrix *dZ, data_t lr) {
    // to reduce number of memory references
    int Ax = ll->A->rows, Ay = ll->A->cols;
    int Wx = ll->W->rows, Wy = ll->W->cols;
    int dZx = dZ->rows, dZy = dZ->cols;
    
    // Allocate dA if not already
    matrix_allocate(ll->dA, Ax, Ay);

    // call forward pass kernel
    // Compute back-propagation error using dZ
    dim3 block_W(BLOCK_SIZE_W, BLOCK_SIZE_W);
    dim3 grid_W((Ax+block_W.x-1)/block_W.x, (Ay+block_W.y-1)/block_W.y);
    FFNNBP_global<<<grid_W, block_W>>>(ll->dA->data_d,
                                        ll->W->data_d,
                                        dZ->data_d,
                                        Wx, Wy,
                                        dZx, dZy);
    
    // update bias
    dim3 block_b(BLOCK_SIZE_b);
    dim3 num_blocks_b((dZy*dZx+block_b.x-1)/block_b.x);
    FFNNUb_global<<<num_blocks_b, block_b>>>(ll->b->data_d,
                                            dZ->data_d,
                                            dZx, dZy,
                                            ll->b->rows,
                                            lr);
    
    // update Weights
    FFNNUW_global<<<grid_W, block_W>>>(ll->W->data_d,
                                        dZ->data_d,
                                        ll->A->data_d,
                                        dZx, dZy,
                                        Ax, Ay,
                                        lr);
    
    copy_matrix_D2H(ll->dA);
    // copy_matrix_D2H(ll->W);
    copy_matrix_D2H(ll->b);

    printf("\n\nLinear Layer Backprop\n");
    print_matrix(ll->dA);
    printf("\n\n");
    printf("\n\nbias\n");
    print_matrix(ll->b);
    printf("\n\n");
    return ll->dA;
}

Matrix getW(Linear_Layer *ll) {
    return *(ll->W);
}

Matrix getb(Linear_Layer *ll) {
    return *(ll->b);
}

int getWx(Linear_Layer *ll) {
    return ll->W->rows;
}

int getWy(Linear_Layer *ll) {
    return ll->W->cols;
}
