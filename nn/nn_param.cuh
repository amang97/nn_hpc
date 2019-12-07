/* Copyright 2019, Aman Gupta, ENG EC 527, Prof. Martin Herbordt              */
/******************************************************************************/
#pragma once

typedef float data_t;

/* Neural Net Parameters */
#define NUM_LAYERS      1 + 1 //+ 1
#define EPOCHS          1
#define LEARNING_RATE   0.01
#define SEED            1527

/* Data Parameters */
#define BATCH_SIZE      3     // num of images fed to Feed Forward NN at once
#define NUM_BATCHES_TR  3
#define NUM_BATCHES_TT  3
#define IMAGE_W         28    // Image width
#define IMAGE_H         28    // Image height
#define BUFFER_SIZE     5120
#define NUM_OUTPUTS     10    // number of inference classes
#define NUM_FEATURES    IMAGE_H * IMAGE_W

/* Kernel Call parameters */
#define BLOCK_SIZE_W    8
#define BLOCK_SIZE_b    256
