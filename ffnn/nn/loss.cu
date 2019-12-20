#include <math.h>
#include <stdio.h>
#include <assert.h>

#include "matrix.cuh"
#include "./../ffnn.cuh"
#include "loss.cuh"

__global__
void cce_loss(data_t * loss, Matrix * Y_pred, Matrix * Y_truth) {

}

data_t compute_cceloss(ffnn * nn, Matrix * Y_pred, Matrix * Y_truth) {
    return (data_t)0;
}

__global__
void BinaryCrossEntropy(data_t *loss, data_t *Y_pred, data_t *Y_truth, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // calculate partial cost (pc)
        data_t pc = (Y_truth[i]*((data_t)log(Y_pred[i]))) + 
            (((data_t)1 - Y_truth[i])*((data_t)log((data_t)1-Y_pred[i])));
        // data_t e = Y_pred[i] - Y_truth[i];
        // data_t pc = 0.5 * e * e;
        atomicAdd(loss,pc/N);
    }
}

__global__
void dBinaryCrossEntropy(data_t *dY, data_t *Y_pred, data_t *Y_truth, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        dY[i] = (data_t)(-1)*(Y_truth[i]/Y_pred[i] - 
                (((data_t)1-Y_truth[i])/((data_t)1-Y_pred[i])));
        // dY[i] = Y_truth[i]-Y_pred[i];
    }
}

data_t BCEloss(Matrix *Y_pred, Matrix *Y_truth) {
    int Ypx = Y_pred->rows, Ytx = Y_truth->rows;
    assert(Ypx == Ytx);
    data_t *BCEloss;

    // unified memory model, because loss read after every epoch
    cudaMallocManaged(&BCEloss, sizeof(data_t));
    * BCEloss = (data_t)0;
    dim3 block_b(BLOCK_SIZE_b);
    dim3 num_blocks_b((Ypx+block_b.x-1)/block_b.x);
    BinaryCrossEntropy<<<num_blocks_b, block_b>>>(BCEloss,
                                                Y_pred->data_d,
                                                Y_truth->data_d,
                                                Ypx);
    
    // synchronize devices before proceeding in main thread
    cudaDeviceSynchronize();
    data_t loss_val = *BCEloss;
    cudaFree(BCEloss);
    
    return loss_val;
}

void dBCEloss(Matrix *dY, Matrix *Y_pred, Matrix *Y_truth) {
    int Ypx = Y_pred->rows, Ytx = Y_truth->rows;
    assert(Ypx == Ytx);
    dim3 block_b(BLOCK_SIZE_b);
    dim3 num_blocks_b((Ypx+block_b.x-1)/block_b.x);
    dBinaryCrossEntropy<<<num_blocks_b, block_b>>>(dY->data_d,
                                                Y_pred->data_d,
                                                Y_truth->data_d,
                                                Ypx*NUM_OUTPUTS);

}
