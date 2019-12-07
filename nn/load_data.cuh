#pragma once

#include "nn_param.cuh"
#include "matrix.cuh"

typedef struct DataTr {
    Matrix *batch[NUM_BATCHES_TR];
    Matrix *label[NUM_BATCHES_TR];
} data_tr;

typedef struct DataTT {
    Matrix *batch[NUM_BATCHES_TT];
    Matrix *label[NUM_BATCHES_TT];
} data_tt;

data_tr *load_mnist_train(char *file);
data_tt *load_mnist_test(char *file);

Matrix *get_batch_data_tr(data_tr *mnist_tr, int batch_id);
Matrix *get_batch_data_tt(data_tt *mnist_tt, int batch_id);

Matrix *get_batch_label_tr(data_tr *mnist_tr, int batch_id);
Matrix *get_batch_label_tt(data_tt *mnist_tt, int batch_id);
