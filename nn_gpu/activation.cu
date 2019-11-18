#include <math.h>
#include "matrix.cuh"

__device__
data_t sigmoid(data_t x) {
    return ((data_t)1) / ((data_t)1 + (data_t)exp(-x));
}

__global__
void sigmoid_activation_forward(data_t *A, data_t *Z, int Zx, int Zy) {
    int i = (blockId.x)*(blockDim.x) + threadId.x;
    int N = Zx * Zy;
    if (i < N) A[i] = sigmoid(Z[i])
}