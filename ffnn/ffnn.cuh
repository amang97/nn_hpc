/* Copyright 2019, Aman Gupta, ENG EC 527, Prof. Martin Herbordt              */
/******************************************************************************/
#pragma once

#include "nn_param.cuh"
#include "nn/matrix.cuh"
#include "nn/nn_layer.cuh"

typedef struct Feed_Forward_Neural_Net {
    nnlayer * layer[NUM_LAYERS];
    Matrix * Y;
} ffnn;

ffnn * ffnn_init();

int ffnn_free(ffnn * nn);

void add_layer(ffnn * nn, int l, int Wx, int Wy, char f);

Matrix * ffnn_fp_global(ffnn * nn, Matrix * X);