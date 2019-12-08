/* Copyright 2019, Aman Gupta, ENG EC 527, Prof. Martin Herbordt              */
/******************************************************************************/
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "ffnn.cuh"
#include "nn_param.cuh"
#include "data/data.cuh"
#include "nn/matrix.cuh"
#include "nn/nn_layer.cuh"
#include "nn/cuda_utils.cuh"

/* Main */
int main() {
    srand(time(0));

    // Set GPU Device
    CUDA_SAFE_CALL(cudaSetDevice(0));

    // load training and testing data
    char * train = (char *)"./data/mnist/mnist_train.csv";
    char * test = (char *)"./data/mnist/mnist_test.csv";
    data_tr *mnist_tr = load_mnist_train(train);
    data_tt *mnist_tt = load_mnist_test(test);
    printf("Training and Test data Loaded\n");

    // creating and adding layers to a feed forward neural net
    ffnn * nn = ffnn_init();
    add_layer(nn, 0, 784, 60, 'r');
    add_layer(nn, 1, 60, 10, 'r');

    /* Network Training on GPU */
    Matrix * Y_pred = matrix_init(BATCH_SIZE, 1); matrix_allocate(Y_pred);
    int epoch, batch;
    for (epoch = 0; epoch < EPOCHS; epoch++) {
        // leave a batch to compute train accuracy later
        for (batch = 0; batch < NUM_BATCHES_TR - 1; batch++) {
            // load data batch
            Matrix * X = get_batch_data_tr(mnist_tr, batch);
            Matrix * Y_truth = get_batch_label_tr(mnist_tr, batch);
            
            // Forward Pass
            Y_pred = ffnn_fp_global(nn, X);

            // Back Propagation

        }
    }
    
    /* Compute train accuracy */


    /* free the neural net (all the layers included)*/
    int n = ffnn_free(nn); if (n) return -1;

    return 0;
}
