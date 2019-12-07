/* Copyright 2019, Aman Gupta, Boston University */
#include "neural_network.cuh"
#include "linear_layer.cuh"
#include "matrix.cuh"
#include <stdio.h>
#include <stdlib.h>
#include "loss.cuh"
#include <string.h>
#include "cuda_utils.cuh"

Neural_Network *neural_net_init() {
    int l;
    Neural_Network *nn = (Neural_Network *)malloc(sizeof(Neural_Network));
    for (l = 0; l < NUM_LAYERS; l++) {
        nn->layer[l] = (Linear_Layer *)malloc(sizeof(Linear_Layer));
        if (!nn->layer[l]) {printf("Unable to allocate layers\n"); return NULL;}
        nn->activation[l] = (Activation_Layer *)malloc(sizeof(Activation_Layer));
        nn->layer_name[l] = (char *)"";
    }
    return nn;
}

void add_layer(Neural_Network *nn, int l, char *ln, int Ax, int Ay, int Wx, int Wy) {
    if (!nn) { printf("No neural net found\n"); exit(EXIT_FAILURE); }
    ll_init(nn->layer[l], Ax, Ay, Wx, Wy);
    nn->layer_name[l] = (char *)ln;
}

void add_activation(Neural_Network *nn, int l, char*An) {
    if (!nn) { printf("No neural net found\n"); exit(EXIT_FAILURE); }
    nn->activation[l]->r = NULL; nn->activation[l]->s = NULL;
    if (!strcmp(An,(char*)"Relu")) {
        printf("Hi from Relu to layer %d\n", l);
        nn->activation[l]->r = relu_activate(nn->layer[l]);
    } else if (!strcmp(An,(char*)"Sigmoid")) {
        printf("Hi from Sigmoid to layer %d\n", l);
        nn->activation[l]->s = sigmoid_activate(nn->layer[l]);
    }

}


Matrix *nn_forward_pass_global(Neural_Network *nn, Matrix *X) {
    Matrix *Z = X;
    //print_matrix(Z);
    int l;
    for (l = 0; l < NUM_LAYERS; l++) {
        printf("Layer: %d\n", l);
        printf("layer size:\n - Input: %d X %d\n - Output: %d X %d\n - Weights: %d X %d\n",
            nn->layer[l]->A->rows, nn->layer[l]->A->cols, nn->layer[l]->Z->rows, nn->layer[l]->Z->cols,
            nn->layer[l]->W->rows,nn->layer[l]->W->cols);
        Z = ll_forward_pass_global(nn->layer[l], Z);
        Relu *r = nn->activation[l]->r;
        Sigmoid *s = nn->activation[l]->s;
        if (r)
            Z = relu_forward_pass_global(r, Z);
        if (s)
            Z = sigmoid_forward_pass_global(s, Z);
        if ((!r) && (!s)) { printf("Forgot to activate the layer\n"); }
        //print_matrix(Z);
        //printf("\n");
    }

    Matrix *Y = Z;
    return Y;
}

void nn_backward_prop_global(Neural_Network *nn, Matrix *Y_pred, Matrix *Y_truth, data_t lr) {
    Matrix *dY = matrix_init(Y_pred->rows, Y_pred->cols);
    matrix_allocate(dY, Y_pred->rows, Y_pred->cols);
    Matrix *bce_error = dBCEloss(dY, Y_pred, Y_truth);

    int l;
    for (l = NUM_LAYERS-1; l >= 0; l--) {        
        Relu *r = nn->activation[l]->r;
        Sigmoid *s = nn->activation[l]->s;
        if (r) {
            bce_error = relu_back_propagation_global(r, bce_error, lr);
        }
        if (s) {
            bce_error = sigmoid_back_propagation_global(s, bce_error, lr);
        }
        if ((!r) && (!s)) { printf("Forgot to activate the layer\n"); }
        
        bce_error = ll_back_propagation_global(nn->layer[l], bce_error, lr);

    }
    // once we finish back propagation, we need to halt main thread execution
    // for all threads to finish
    cudaDeviceSynchronize(); 
}
