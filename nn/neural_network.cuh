/* Copyright 2019, Aman Gupta, Boston University */
#pragma once

#include "nn_param.cuh"
#include "linear_layer.cuh"
#include "matrix.cuh"

typedef struct Neural_Network {
    Linear_Layer *layer[NUM_LAYERS];
    char * layer_name[NUM_LAYERS];
} Neural_Network;

Neural_Network *neural_net_init();

void add_layer(Neural_Network *nn, int l, char *ln, int Ax, int Ay, int Wx, int Wy, int seed);

Matrix *nn_forward_pass_global(Neural_Network *nn, Matrix *X);
