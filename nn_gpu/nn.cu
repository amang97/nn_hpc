#include "matrix.cuh"
#include "nnlayer.cuh"
#include "nn.cuh"
#include "cost.cuh"

matrix& NeuralNetwork_ForwardPass_global(matrix& X, layer &l[], int num_layers) {
    matrix Z = X;
    int layer;
    for (layer = 0; layer < num_layers; layer++) {
        forward_pass_global(&l[layer], Z, Z->rows, Z->cols);
    }
    matrix Y = Z;
    return Y;
}

void accuracy(const matrix& pred, const matrix& Y) {
    int n = Y->rows*Y->cols;
    int i; data_t error_h = (data_t)0; data_t error_h = (data_t)0;
    for (i = 0; i < n; i++)
        if (pred->data_h[i] != Y->data_h[i]) error_h += (data_t)1;
        if (pred->data_d[i] != Y->data_d[i]) error_d += (data_t)1;
    print("error_d: %f, error_h: %f\n", error_d, error_h);
}

// matrix& NeuralNetwork_BackProp_global(matrix& pred, matrix& Y, layer &l[], int num_layers) {
//     matrix *dY = matrix_allocate(Y->rows, Y->cols);
//     matrix error = 
// }