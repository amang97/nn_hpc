#pragma once

#include "nn_param.cuh"
#include "matrix.cuh"

__global__
void BinaryCrossEntropy(data_t *loss, data_t *Y_pred, data_t *Y_truth, int N);

__global__
void dBinaryCrossEntropy(data_t *dY, data_t *Y_pred, data_t *Y_truth, int N);

data_t BCEloss(Matrix *Y_pred, Matrix *Y_truth);

Matrix * dBCEloss(Matrix *dY, Matrix *Y_pred, Matrix *Y_truth);

data_t accuracy(const Matrix * Y_pred, const Matrix * Y_truth);
