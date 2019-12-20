all: main.cu data/data.cu data/data.cuh nn/matrix.cu nn/matrix.cuh nn/cuda_utils.cu nn/cuda_utils.cuh nn/nn_layer.cu nn/nn_layer.cuh nn/activations.cuh nn/activations.cu nn/loss.cuh nn/loss.cu ffnn.cu ffnn.cuh nn_param.cuh; module load cuda; nvcc ./data/data.cu ./nn/matrix.cu ./nn/cuda_utils.cu ./nn/nn_layer.cu ./nn/activations.cu ./nn/loss.cu ffnn.cu main.cu -o main; clear;
clean: main; rm main;
