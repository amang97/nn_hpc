all: main.cu cuda_utils.cu cuda_utils.cuh matrix.cu matrix.cuh linear_layer.cu linear_layer.cuh load_data.cu load_data.cuh neural_network.cu neural_network.cuh loss.cuh nn_param.cuh cuPrintf.cuh; module load cuda; nvcc cuda_utils.cu matrix.cu linear_layer.cu load_data.cu neural_network.cu main.cu -o main;
clean: rm main;
