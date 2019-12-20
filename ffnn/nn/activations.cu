#include "matrix.cuh"
#include "nn_layer.cuh"
#include "cuda_utils.cuh"
#include "../nn_param.cuh"
#include "activations.cuh"
/******************************************************************************/
/* Activations */
/******************************************************************************/
__device__
data_t relu(data_t x, data_t y) {
    return (x > y) ? x : y;
}

__device__
data_t sigmoid(data_t x) {
    return data_t((((data_t)1) / ((data_t)1 + (data_t)exp(-x)))/NUM_OUTPUTS);
}

__global__
void relu_forward_global(data_t *A, data_t *Z, int Zx, int Zy) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < Zx * Zy) {
        A[index] = relu(Z[index],(data_t)0);
    }
}

__global__
void sigmoid_forward_global(data_t *A, data_t *Z, int Zx, int Zy) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < Zx * Zy) {
        A[index] = sigmoid(Z[index]);
    }
}

/* RELU Activation Backward Pass*/
/******************************************************************************/
__global__
void relu_backward_global(data_t *dZ, data_t *dA, data_t *Z, int Zx, int Zy) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < Zx * Zy) {
        dZ[index] = (Z[index] > 0) ? dA[index] : 0;
    }
}

/* Sigmoid Activation Backward Pass*/
/******************************************************************************/
__global__
void sigmoid_backward_global(data_t *dZ, data_t *dA, data_t *Z, int Zx, int Zy) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < Zx*Zy) {
        dZ[index] = dA[index]*sigmoid(Z[index])*((data_t)1 - sigmoid(Z[index]));
    }
}

/* Host calls to GPU for RELU for forward pass*/
void relu_forward_pass_global(Matrix * A, Matrix * Z) {
    int  Zx = Z->rows, Zy = Z->cols;
    // call relu activation forward pass
    dim3 block(BLOCK_SIZE_b);
    dim3 num_blocks((Zy*Zx+block.x-1)/block.x);
    relu_forward_global<<<num_blocks,block>>>(A->data_d,
                                            Z->data_d,
                                            Zx, Zy);
    // return A;
}

/* Host calls to GPU for Sigmoid for Forward pass */
void sigmoid_forward_pass_global(Matrix * A, Matrix * Z) {
    int Zx = Z->rows; int Zy = Z->cols;

    // call sigmoid activation forward pass
    dim3 block(BLOCK_SIZE_b);
    dim3 num_blocks((Zy*Zx+block.x-1)/block.x);
    sigmoid_forward_global<<<num_blocks,block>>>(A->data_d,
                                                Z->data_d,
                                                Zx, Zy);
    // return A;
}

/* Host calls to GPU for RELU for backProp */
void relu_back_propagation_global(nnlayer * nnl, Matrix *dA, data_t lr) {
    int Zx = nnl->Z->rows; int Zy = nnl->Z->cols;
    dim3 block(BLOCK_SIZE_b);
    dim3 num_blocks((Zy*Zx+block.x-1)/block.x);
    relu_backward_global<<<num_blocks,block>>>(nnl->dZ->data_d,
                                                dA->data_d,
                                                nnl->Z->data_d,
                                                Zx, Zy);
    // return nnl->dZ;
}

/* Host calls to GPU for Sigmoid for backprop*/
void sigmoid_back_propagation_global(nnlayer * nnl, Matrix *dA, data_t lr) {
    int Zx = nnl->Z->rows; int Zy = nnl->Z->cols;
    dim3 block(BLOCK_SIZE_b);
    dim3 num_blocks((Zy*Zx+block.x-1)/block.x);
    sigmoid_backward_global<<<num_blocks,block>>>(nnl->dZ->data_d,
                                                dA->data_d,
                                                nnl->Z->data_d,
                                                Zx, Zy);
    // return nnl->dZ;
}

