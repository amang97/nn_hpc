/* Copyright 2019, Aman Gupta, Boston University */
#pragma once

#include "nn_param.cuh"
#include "linear_layer.cuh"
#include "matrix.cuh"
#include "activation_layer.cuh"

typedef struct Neural_Network {
    Linear_Layer *layer[NUM_LAYERS];
    Activation_Layer *activation[NUM_LAYERS];
    char * layer_name[NUM_LAYERS];
} Neural_Network;

Neural_Network *neural_net_init();

void add_layer(Neural_Network *nn, int l, char *ln, int Ax, int Ay, int Wx, int Wy);
void add_activation(Neural_Network *nn, int l, char*An);

Matrix *nn_forward_pass_global(Neural_Network *nn, Matrix *X);

void nn_backward_prop_global(Neural_Network *nn, Matrix *Y_pred, Matrix *Y_truth, data_t lr);
