#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "nnlayer.cuh"        // bsize_b, bsize_W
#include "activation.cuh"     // bsize_b
#include "matrix.cuh"         // data_t
#include "data.cuh"
#include "nn.cuh"

/* Neural Net Parameters */
#define NUM_LAYERS      3
#define EPOCHS          100
#define LEARNING_RATE   0.01
#define SEED            1527
/* Data Parameters */
#define NUM_INPUTS      100   // total data_points (must be multiple of batch_size)
#define BATCH_SIZE      100   // num of input data_points
#define IMAGE_W         28    // Image width
#define IMAGE_H         28    // Image height
#define NUM_OUTPUTS     10    // number of inference classes
#define NUM_FEATURES    IMAGE_H * IMAGE_W

// Assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), (char *)__FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "CUDA_SAFE_CALL: %s %s %d\n",
                                       cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

int main() {
    srand(time(NULL));

    // load training and testing data
    char * train = (char *)"./mnist/mnist_train.csv";
    char * test = (char *)"./mnist/mnist_test.csv";
    dataset *mnist_train = read_batch(0, BATCH_SIZE, IMAGE_W, IMAGE_H, train);
    if (!mnist_train) {printf("Couldn't load train data"); return -1;}
    dataset *mnist_test = read_batch(0, BATCH_SIZE, IMAGE_W, IMAGE_H, test);
    if (!mnist_test) {printf("Couldn't load test data"); return -1;}
    printf("\nTraining and Test data Loaded\n");

    // Create a Feed Forward Neural Net (array of layers) and other parameters
    layer l[NUM_LAYERS];
    data_t lr = LEARNING_RATE;

    // Set GPU Device
    CUDA_SAFE_CALL(cudaSetDevice(0));

    // initialize layers
    layer_init(l[0], BATCH_SIZE, NUM_FEATURES, NUM_FEATURES, 60, SEED);
    layer_init(l[1], NUM_FEATURES, 60, 60, NUM_OUTPUTS, SEED);
    layer_init(l[2], 60, NUM_OUTPUTS, NUM_OUTPUTS, 1, SEED);
    printf("\nNeural Net layers initialized\n");
    
    // Network Training
    matrix pred;
    int epoch;//, batch, num_batches = (int)(NUM_INPUTS/BATCH_SIZE);
    matrix *X = matrix_allocate(BATCH_SIZE, NUM_FEATURES);
    matrix *Y = matrix_allocate(BATCH_SIZE, 1);
    X->data_h = mnist_train->images; Y->data_h = mnist_train->labels;
    CUDA_SAFE_CALL(cudaMemcpy(X->data_d, X->data_h, BATCH_SIZE*NUM_FEATURES*sizeof(data_t),
                                                    cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(Y->data_d, Y->data_h, BATCH_SIZE*1*sizeof(data_t),
                                                    cudaMemcpyHostToDevice));
      for (epoch = 0; epoch < EPOCHS; epoch++) {
      data_t cost = (data_t)0;
      // for (batch = 0; batch < num_batches; batch++) {
      //   Y = NeuralNetwork_ForwardPass_global(batch_data);
      // }
      pred = NeuralNetwork_ForwardPass_global(X, l, NUM_LAYERS);
      accuracy(pred,Y);
    }


    // delete layers
    int l1 = delete_layer(l[0]);
    int l2 = delete_layer(l[1]);
    int l3 = delete_layer(l[2]);
    if (!(l1) || !(l2) || !(l3)) printf("Neural Network layers destroyed\n");
    else printf("OOpS!, Neural Net destruction went wrong.\n");
    
    return 0;
}
