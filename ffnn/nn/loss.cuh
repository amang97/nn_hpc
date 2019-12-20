#pragma once

#include "matrix.cuh"
#include "./../ffnn.cuh"
#include "./../nn_param.cuh"


__global__
void cce_loss(data_t * loss, Matrix * Y_pred, Matrix * Y_truth);

data_t compute_cceloss(ffnn * nn, Matrix * Y_pred, Matrix * Y_truth);


__global__
void BinaryCrossEntropy(data_t *loss, data_t *Y_pred, data_t *Y_truth, int N);

__global__
void dBinaryCrossEntropy(data_t *dY, data_t *Y_pred, data_t *Y_truth, int N);

data_t BCEloss(Matrix *Y_pred, Matrix *Y_truth);

void dBCEloss(Matrix *dY, Matrix *Y_pred, Matrix *Y_truth);