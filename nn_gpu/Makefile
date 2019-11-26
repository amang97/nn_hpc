all: main.cu matrix.cu nnlayer.cu activation.cu data.cu matrix.cuh nnlayer.cuh activation.cuh data.cuh; module load cuda; nvcc main.cu matrix.cu nnlayer.cu activation.cu data.cu -o main;
clean: main; rm main;
