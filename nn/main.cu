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
#include "activation_layer.cuh"

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
    char *Relu = (char *)"Relu";
    char *Sigmoid = (char *)"Sigmoid";
    add_layer(nn, 0, (char *)"Input Layer", 3, 784, 784, 60);
    add_activation(nn, 0, Relu);
    add_layer(nn, 1, (char *)"Hidden Layer", 3, 60, 60, 10);
    add_activation(nn, 1, Sigmoid);

    // // initialization verification
    // int l;
    // for (l = 0; l < NUM_LAYERS; l++) {
    //     printf("Layer: %d, Name: %s\n", l, nn->layer_name[l]);
    //     printf("\n\nW: %d X %d\n", nn->layer[l]->W->rows, nn->layer[l]->W->cols);
    //     printf("Host allocation: %s, device allocation: %s", nn->layer[l]->W->host_assigned, nn->layer[l]->W->device_assigned);
    //     print_matrix(nn->layer[l]->W);
    //     printf("\n");
    //     print_matrix_d(nn->layer[l]->W);
    //     printf("\n");

    // }
    //add_layer(nn, 2, (char *)"Output Layer", 10, 1, 1234);
    
    /* Network Training on GPU */
    Matrix *Y_pred = matrix_init(BATCH_SIZE, 1);
    matrix_allocate(Y_pred, BATCH_SIZE, 1);
    int epoch, batch;
    for (epoch = 0; epoch < EPOCHS; epoch++) {
        data_t loss = (data_t)0;
        printf("\n\nepoch: %d\n", epoch);
        for (batch = 0; batch < NUM_BATCHES_TR - 1; batch++) {
            printf("batch: %d\n", batch);
            Matrix *X = get_batch_data_tr(mnist_tr, batch);
            Matrix *Y_truth = get_batch_label_tr(mnist_tr, batch);
            printf("Xsize: %d X %d; Ysize: %d X %d\n", X->rows, X->cols, Y_truth->rows, Y_truth->cols);
            Y_pred = nn_forward_pass_global(nn, X);
            //printf("\nForward Pass Y Prediction:-\n");
            //copy_matrix_D2H(Y_pred);
            //print_matrix(Y_pred);
            nn_backward_prop_global(nn, Y_pred, Y_truth, LEARNING_RATE);
            printf("\nBack Propagation\n");
            //copy_matrix_D2H(Y);
            loss += BCEloss(Y_pred, Y_truth);
        }
        printf("Loss at Epoch %d: %f\n", epoch, loss);
    }
    
    /* Compute train accuracy */

    // // delete layers
    // if (!(l1) || !(l2) || !(l3)) printf("Neural Network layers destroyed\n");
    // else printf("OOpS!, Neural Net destruction went wrong.\n");
    
    return 0;
}
