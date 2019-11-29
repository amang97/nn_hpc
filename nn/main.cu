/* Copyright 2019, Aman Gupta, ENG EC 527, Prof. Martin Herbordt              */
/******************************************************************************/
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "matrix.cuh"
#include "cuda_utils.cuh"
#include "nn_param.cuh"
#include "linear_layer.cuh"
#include "loss.cuh"
#include "load_data.cuh"
#include "neural_network.cuh"

/* Main */
int main() {
    srand(time(0));

    // Set GPU Device
    CUDA_SAFE_CALL(cudaSetDevice(0));

    // load training and testing data
    char * train = (char *)"./mnist/mnist_train.csv";
    char * test = (char *)"./mnist/mnist_test.csv";
    data_tr *mnist_tr = load_mnist_train(train);
    data_tt *mnist_tt = load_mnist_test(test);
    printf("\nTraining and Test data Loaded\n");

    // Create a Feed Forward Neural Net (array of layers) and other parameters
    Neural_Network *nn = neural_net_init();
    printf("Neural Network layers allocated\n");


    // initialize network layer weights and biases
    add_layer(nn, 0, (char *)"Input Layer", 3, 784, 784, 60, 1234);
    add_layer(nn, 1, (char *)"Hidden Layer", 784, 60, 60, 10, 1234);
    //add_layer(nn, 2, (char *)"Output Layer", 10, 1, 1234);
    
    /* Network Training on GPU */
    Matrix *Y = matrix_init(BATCH_SIZE, 1);
    matrix_allocate(Y, BATCH_SIZE, 1);
    int epoch, batch;
    for (epoch = 0; epoch < EPOCHS; epoch++) {
        // data_t loss = (data_t)0;
        printf("epoch: %d\n", epoch);
        for (batch = 0; batch < NUM_BATCHES_TR - 1; batch++) {
            printf("batch: %d\n", batch);
            Matrix *X = get_batch_data_tr(mnist_tr, batch);
            printf("Xsize: %d\n", X->rows*X->cols);
            //print_matrix(X);
            Y = nn_forward_pass_global(nn, X);
            // loss += BCEloss(Y, get_batch_label_tr(mnist_tr, batch));
        }
    }

    
    /* Compute train accuracy */

    // // delete layers
    // if (!(l1) || !(l2) || !(l3)) printf("Neural Network layers destroyed\n");
    // else printf("OOpS!, Neural Net destruction went wrong.\n");
    
    return 0;
}
