#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "matrix.cuh"
#include "cuda_utils.cuh"
#include "nn_param.cuh"

/* Main */
int main() {
    srand(time(NULL));

    // // load training and testing data
    // char * train = (char *)"./mnist/mnist_train.csv";
    // char * test = (char *)"./mnist/mnist_test.csv";

    // printf("\nTraining and Test data Loaded\n");

    // Set GPU Device
    CUDA_SAFE_CALL(cudaSetDevice(0));
    
    // initialize matrices
    Matrix *W = matrix_init(100,100);
    matrix_allocate(W,100,100);
    printf("Weights initializedon device and host")
    print_matrix(W);
    int w = matrix_free(W);
    if (!w) printf("Weights freed")
    // // Create a Feed Forward Neural Net (array of layers) and other parameters
    // layer l[NUM_LAYERS];
    // data_t lr = LEARNING_RATE;

    // // initialize layers
    // layer_init(l[0], BATCH_SIZE, NUM_FEATURES, NUM_FEATURES, 60, SEED);
    // layer_init(l[1], NUM_FEATURES, 60, 60, NUM_OUTPUTS, SEED);
    // layer_init(l[2], 60, NUM_OUTPUTS, NUM_OUTPUTS, 1, SEED);
    // printf("\nNeural Net layers initialized\n");
    
    /* Network Training on GPU */

    // // delete layers
    // int l1 = delete_layer(l[0]);
    // int l2 = delete_layer(l[1]);
    // int l3 = delete_layer(l[2]);
    // if (!(l1) || !(l2) || !(l3)) printf("Neural Network layers destroyed\n");
    // else printf("OOpS!, Neural Net destruction went wrong.\n");
    
    return 0;
}
