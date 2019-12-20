#include <stdio.h>
#include <assert.h>
#include "ffnn.cuh"
#include "nn/loss.cuh"
#include "nn_param.cuh"
#include "nn/nn_layer.cuh"
#include "nn/cuda_utils.cuh"
#include "nn/activations.cuh"

ffnn * ffnn_init() {
    ffnn * nn = (ffnn *)malloc(sizeof(ffnn));
    if (!nn) { printf("Unable to initialize ffnn\n"); return NULL; }

    int l;
    for (l = 0; l < NUM_LAYERS; l++) {
        nn->layer[l] = (nnlayer *)malloc(sizeof(nnlayer));
    }
    nn->Y = matrix_init(BATCH_SIZE, NUM_OUTPUTS);
    nn->dY = matrix_init(BATCH_SIZE, NUM_OUTPUTS);
    matrix_allocate(nn->Y);
    matrix_allocate(nn->dY);
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
    matrix_free(nn->dY);
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
    int l;
    for (l = 0; l < NUM_LAYERS-1; l++) {
        nn->layer[l]->Z = nnl_forward_pass_global(nn->layer[l], nn->layer[l]->A);
        switch (nn->layer[l]->f) {  // activation of current hidden layer
            // activate Z of cur layer and forward pass as input of next layer
            case 'r': relu_forward_pass_global(nn->layer[l+1]->A, nn->layer[l]->Z); break;
            case 's': sigmoid_forward_pass_global(nn->layer[l+1]->A, nn->layer[l]->Z); break;
            default: break;
        }
    }

    // Forward Pass through output layer to get Prediction matrix Y
    nn->layer[NUM_LAYERS-1]->Z = nnl_forward_pass_global(nn->layer[NUM_LAYERS-1], nn->layer[NUM_LAYERS-1]->A);
    switch (nn->layer[NUM_LAYERS-1]->f) {  // activation of Output layer
        case 'r': relu_forward_pass_global(nn->Y, nn->layer[NUM_LAYERS-1]->Z); break;
        case 's': sigmoid_forward_pass_global(nn->Y, nn->layer[NUM_LAYERS-1]->Z); break;
        default: break;
    }
    return nn->Y;
}

// back propagation (bp) through the layers of the neural net
void ffnn_bp_global(ffnn * nn, Matrix * Y_pred, Matrix * Y_truth, data_t lr) {
    dBCEloss(nn->dY, Y_pred, Y_truth);
    // uncomment below to see loss at each epoch for each batch, recommended Batch-size =10, #epoch = 3 to 5
    // printf("Loss\n");
    // print_matrix_d(nn->dY);
    // printf("\n");

    // Back Prop through output layer
    switch (nn->layer[NUM_LAYERS-1]->f) { 
        case 'r': relu_back_propagation_global(nn->layer[NUM_LAYERS-1], nn->dY, lr); break;
        case 's': sigmoid_back_propagation_global(nn->layer[NUM_LAYERS-1], nn->dY, lr); break;
        default: break;
    }
    nnl_back_propagation_global(nn->layer[NUM_LAYERS-1], nn->layer[NUM_LAYERS-1]->dZ, lr);
    // uncomment below to see weight and bias updates for last layer
    // printf("Weights\n");
    // print_matrix_d(nn->layer[NUM_LAYERS-1]->W);
    // printf("\n");
    // printf("bias\n");
    // print_matrix_d(nn->layer[NUM_LAYERS-1]->b);
    // printf("\n");

    // Back propagate through hidden linear
    int l;
    for (l = NUM_LAYERS-1; l > 0; l--) {
        switch (nn->layer[l]->f) {  // activation of current hidden layer
            case 'r': relu_back_propagation_global(nn->layer[l-1], nn->layer[l]->dA, lr); break;
            case 's': sigmoid_back_propagation_global(nn->layer[l-1], nn->layer[l]->dA, lr); break;
            default: break;
        }
        nnl_back_propagation_global(nn->layer[l-1], nn->layer[l-1]->dZ, lr);
        // uncomment below to see weight and bias updates for all layers (Not recommended)
        // printf("Weights\n");
        // print_matrix_d(nn->layer[NUM_LAYERS-1]->W);
        // printf("\n");
        // printf("bias\n");
        // print_matrix_d(nn->layer[NUM_LAYERS-1]->b);
        // printf("\n");
    }

    // need device synchronize because next iteration forward pass input depends on output
    // from this iterations back propagation
    cudaDeviceSynchronize();
}

data_t * get_prediction(Matrix * Y_batch) {
    if (!Y_batch) return NULL;
    data_t * y_label = (data_t *)calloc(BATCH_SIZE, sizeof(data_t));
    int i, j;
    for (i = 0; i < BATCH_SIZE; i++) {
        data_t max = Y_batch->data_h[i*NUM_OUTPUTS];
        for (j = 0; j < NUM_OUTPUTS; j++) {
            data_t e = Y_batch->data_h[i*NUM_OUTPUTS+j];
            if (e >= max) {
                max = e;
                y_label[i] = j;
            }
        }
    }
    return y_label;
}

data_t get_accuracy(Matrix * Y_pred, Matrix * Y_truth) {
    data_t * Y_label_tr = get_prediction(Y_truth);
    copy_matrix_D2H(Y_pred);
    data_t * Y_label_pr = get_prediction(Y_pred);
    int acc, i;
    for (i = 0; i < BATCH_SIZE; i++) {
        //printf("Pred: %d, Truth: %d\n", Y_label_pr[i], Y_label_tr[i]);
        if (Y_label_tr[i] == Y_label_pr[i]) acc++;
    }
    return (data_t)acc/BATCH_SIZE;
}

data_t compute_accuracy(data_t * Y_tr, data_t * Y_pr) {
    int acc = 0, i;
    for (i = 0; i < BATCH_SIZE; i++)
        if (Y_tr[i] == Y_pr[i]) acc++;
    // printf("acc: %d\n", acc);
    return ((data_t)acc/(data_t)BATCH_SIZE);
}
