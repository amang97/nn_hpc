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
#include "nn/activations.cuh"
#include "nn/loss.cuh"

/* Main */
int main() {
    srand(time(0));

    // Set GPU Device
    CUDA_SAFE_CALL(cudaSetDevice(0));

    // load training and testing data by batches, labels loaded 1 hot encoded
    char * train = (char *)"./data/mnist/mnist_train.csv";
    char * test = (char *)"./data/mnist/mnist_test.csv";
    data_tr *mnist_tr = load_mnist_train(train);
    data_tt *mnist_tt = load_mnist_test(test);
    printf("Training and Test data Loaded\n");

    // creating and adding layers to a feed forward neural net
    ffnn * nn = ffnn_init();
    add_layer(nn, 0, 784, 20, 'r');
    // add_layer(nn, 1, 120, 60, 'r');
    add_layer(nn, 1, 20, 10, 's');  // Last layer needs to be sigmoid

    /* Network Training on GPU */
    Matrix * Y_pred = matrix_init(BATCH_SIZE, 1); matrix_allocate(Y_pred);
    int epoch, batch;
    for (epoch = 0; epoch < EPOCHS; epoch++) {
        printf("\nepoch: %d\n", epoch);
        data_t epoch_accuracy = (data_t)0;
        // leave a batch to compute train accuracy later
        for (batch = 0; batch < NUM_BATCHES_TR - 1; batch++) {
            // load data batch
            Matrix * X = get_batch_data_tr(mnist_tr, batch);
            Matrix * Y_truth = get_batch_label_tr(mnist_tr, batch);
 
            // Forward Pass
            Y_pred = ffnn_fp_global(nn, X);
            // printf("Pred: \n");
            // print_matrix_d(Y_pred);
            // printf("Truth: \n");
            // print_matrix_d(Y_truth);

            // Back Propagation
            ffnn_bp_global(nn, Y_pred, Y_truth, LEARNING_RATE);

            // one hot encoding to class label
            data_t * y_label_tr = get_prediction(Y_truth);
            // print_matrix_d(Y_pred);
            copy_matrix_D2H(Y_pred);
            // printf("Prediction:\n");
            // print_matrix(Y_pred);
            data_t * y_label_pr = get_prediction(Y_pred);

            epoch_accuracy += compute_accuracy(y_label_tr, y_label_pr);
        }
        /* Compute train accuracy */
        printf("accuracy = %f\n", epoch, epoch_accuracy/(NUM_BATCHES_TR - 1)); 
    }    

    /* free the neural net (all the layers included)*/
    int n = ffnn_free(nn); if (n) return -1;
    int d = free_data(mnist_tr, mnist_tt); if (d) return -1;
    return 0;
}
