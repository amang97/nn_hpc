all: main.cu matrix.cu nnlayer.cu activation.cu matrix.cuh nnlayer.cuh activation.cuh; module load cuda; nvcc main.cu matrix.cu nnlayer.cu activation.cu -o main;
clean: main; rm main;
