#include <stdio.h>
#include <assert.h>
#include "ffnn.cuh"
#include "nn_param.cuh"
#include "nn/nn_layer.cuh"

ffnn * ffnn_init() {
    ffnn * nn = (ffnn *)malloc(sizeof(ffnn));
    if (!nn) { printf("Unable to initialize ffnn\n"); return NULL; }

    int l;
    for (l = 0; l < NUM_LAYERS; l++) {
        nn->layer[l] = (nnlayer *)malloc(sizeof(nnlayer));
    }
    nn->Y = matrix_init(BATCH_SIZE, NUM_OUTPUTS);
    matrix_allocate(nn->Y);
    return nn;
}

int ffnn_free(ffnn * nn) {
    if (!nn) { printf("Unable to initialize ffnn\n"); return -1; }
    int l;
    for (l = 0; l < NUM_LAYERS; l++) {
        int n = nnl_free(nn->layer[l]);
        if (n) return -1;
    }
    matrix_free(nn->Y);
    free(nn);
    return 0;
}

void add_layer(ffnn * nn, int l, int Wx, int Wy, char f) {
    if (!nn) { printf("Unable to initialize ffnn\n"); exit(1); }
    nn->layer[l] = nnl_init(l, Wx, Wy, f);
}

// Forward pass (fp) through layers of the neural net
Matrix * ffnn_fp_global(ffnn * nn, Matrix * X) {
    nn->layer[0]->A = X;

    // Forward Pass though layers
    Matrix * Z; int l;
    for (l = 0; l < NUM_LAYERS-1; l++) {
        Z = nnl_forward_pass_global(nn->layer[l], nn->layer[l]->A);
        switch (nn->layer[l]->f) {  // activation of current hidden layer
            // activate Z of cur layer and forward pass as input of next layer
            case 'r': relu_forward_pass_global(nn->layer[l+1]->A, Z); break;
            case 's': sigmoid_forward_pass_global(nn->layer[l+1]->A, Z); break;
            default: break;
        }
    }

    // Forward Pass through output layer to get Prediction matrix Y
    Z = nnl_forward_pass_global(nn->layer[NUM_LAYERS-1], nn->layer[NUM_LAYERS-1]->A);
    switch (nn->layer[NUM_LAYERS-1]->f) {  // activation of Output layer
        case 'r': relu_forward_pass_global(nn->Y, Z); break;
        case 's': sigmoid_forward_pass_global(nn->Y, Z); break;
        default: break;
    }
    return nn->Y;
}
