all: main.cu cuda_utils.cu cuda_utils.cuh matrix.cu matrix.cuh load_data.cu load_data.cuh nn_param.cuh; module load cuda; nvcc cuda_utils.cu matrix.cu load_data.cu main.cu -o main;
clean: rm main;
