/* Copyright 2019, Aman Gupta, ENG EC 527, Prof. Martin Herbordt              */
/******************************************************************************/
#pragma once

#include "nn_param.cuh"
#include "nn/matrix.cuh"
#include "nn/nn_layer.cuh"

typedef struct Feed_Forward_Neural_Net {
    nnlayer * layer[NUM_LAYERS];
    Matrix * Y, * dY;
} ffnn;

ffnn * ffnn_init();

int ffnn_free(ffnn * nn);

void add_layer(ffnn * nn, int l, int Wx, int Wy, char f);

Matrix * ffnn_fp_global(ffnn * nn, Matrix * X);

void ffnn_bp_global(ffnn * nn, Matrix * Y_pred, Matrix * Y_truth, data_t lr);


data_t * get_prediction(Matrix * Y_batch);

data_t get_accuracy(Matrix * Y_pred, Matrix * Y_truth);

data_t compute_accuracy(data_t * Y_tr, data_t * Y_pr);