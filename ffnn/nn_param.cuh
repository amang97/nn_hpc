/* Copyright 2019, Aman Gupta, ENG EC 527, Prof. Martin Herbordt              */
/******************************************************************************/
#pragma once

typedef float data_t;

/* Neural Net Parameters */
#define NUM_LAYERS      1 + 1
#define EPOCHS          20
#define LEARNING_RATE   0.1

/* Data Parameters */
#define BATCH_SIZE      60     // num of images fed to Feed Forward NN at once
#define NUM_BATCHES_TR  1000
#define NUM_BATCHES_TT  15
#define IMAGE_W         28    // Image width
#define IMAGE_H         28    // Image height
#define BUFFER_SIZE     5120
#define NUM_OUTPUTS     10    // number of inference classes
#define NUM_FEATURES    IMAGE_H * IMAGE_W

/* Kernel Call parameters */
#define BLOCK_SIZE_W    16
#define BLOCK_SIZE_b    1024