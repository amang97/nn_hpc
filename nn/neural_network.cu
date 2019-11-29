/* Copyright 2019, Aman Gupta, Boston University */
#include "neural_network.cuh"
#include "linear_layer.cuh"
#include "matrix.cuh"
#include <stdio.h>
#include <stdlib.h>

Neural_Network *neural_net_init() {
    int l;
    Neural_Network *nn = (Neural_Network *)malloc(sizeof(Neural_Network));
    for (l = 0; l < NUM_LAYERS; l++) {
        nn->layer[l] = (Linear_Layer *)malloc(sizeof(Linear_Layer));
        if (!nn->layer[l]) {printf("Unable to allocate layers\n"); return NULL;}
        nn->layer_name[l] = (char *)"";
    }
    return nn;
}

void add_layer(Neural_Network *nn, int l, char *ln, int Ax, int Ay, int Wx, int Wy, int seed) {
    if (!nn) { printf("No nneural net found\n"); exit(EXIT_FAILURE); }
    ll_init(nn->layer[l], Ax, Ay, Wx, Wy, seed);
    nn->layer_name[l] = (char *)ln;
}

Matrix *nn_forward_pass_global(Neural_Network *nn, Matrix *X) {
    Matrix *Z = X;
    //print_matrix(Z);
    int l;
    for (l = 0; l < NUM_LAYERS; l++) {
        printf("Layer: %d\n", l);

        Z = ll_forward_pass_global(nn->layer[l], Z);
        //print_matrix(Z);
        //printf("\n");
    }

    Matrix *Y = copy_matrix(Z);
    return Y;
}
