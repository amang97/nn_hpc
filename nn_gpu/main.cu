#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "nnlayer.cuh"      // bsize_b, bsize_W
#include "activation.cuh"   // bsize_b
#include "matrix.cuh"       // data_t

#define NUM_LAYERS      3
#define EPOCHS          1
#define LEARNING_RATE   0.01

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
    layer layers[NUM_LAYERS];
    data_t lr = LEARNING_RATE;

    // Set GPU Device
    CUDA_SAFE_CALL(cudaSetDevice(0));

    // initialize layers
    layer_init(layers[0], 1, 3, 3, 2, 0);
    layer_init(layers[1], 3, 2, 2, 1, 0);
    layer_init(layers[2], 2, 1, 1, 1, 0);

    printf("\nHi, nnlayers initialized\n");

    // delete layers
    int l1 = delete_layer(layers[0]);
    int l2 = delete_layer(layers[1]);
    int l3 = delete_layer(layers[2]);
    if (!(l1 || l2 || l3))
        printf("Whoa, nnlayers destroyed\n");
    else
        printf("Something went wrong bruh!\n");
    
    return 0;
}
