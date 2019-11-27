all: main.cu cuda_utils.cu cuda_utils.cuh matrix.cu matrix.cuh linear_layer.cu linear_layer.cuh load_data.cu load_data.cuh loss.cuh nn_param.cuh; module load cuda; nvcc cuda_utils.cu matrix.cu linear_layer.cu load_data.cu main.cu -o main;
clean: rm main;
