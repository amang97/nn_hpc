/* Copyright 2019, Aman Gupta, ENG EC 527, Prof. Martin Herbordt              */
/******************************************************************************/
#pragma once

typedef float data_t;

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
