#pragma once
#include "matrix.cuh"

matrix& NeuralNetwork_ForwardPass_global(matrix& X, layer &l[], int num_layers);
//matrix& NeuralNetwork_BackProp_global(matrix& pred, matrix& Y, layer &l[], int num_layers);

data_t accuracy(const matrix& pred, const matrix& Y);