#pragma once
#include "matrix.cuh"

typedef data_t Pixel;

typedef struct DataSet {
    int n, w, h;
    Pixel *images;
    int *labels;
} dataset;

dataset * read_batch(int start, int batch_size, int w, int h, char *file);
