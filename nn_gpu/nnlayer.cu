/* Copyright 2019, Aman Gupta, ENG EC 527, Prof. Martin Herbordt              */
/******************************************************************************/
/* Neural Network Layer library implementation for CUDA in C                  */
/******************************************************************************/
/* Libraries */
#include <stdio.h>
#include <stdlib.h>
#include "matrix.cuh"
#include "layer.cuh"

/******************************************************************************/
/* Parameters */
#define TILE_WIDTH              32
#define NUM_THREADS_PER_BLOCK   1024

/******************************************************************************/
/* Implementations */
/******************************************************************************/

/* Multi GPU implementations */
/******************************************************************************/

/* Feed Forward NN Forward Pass (FFNNFP) Linear Layer using global memory
Input:  pointer to output data matrix, pointers to input matrix A and B,
        size of matrix A and width of Matrix B, and current GPUid
        Assumes width of A matches height of B; and global memory view
Output: multiplication output saved in location pointed by output data matrix
*/
__global__
void FFNNFP_global(data_t *out, data_t *A, data_t *B, int Ax, int Ay, int By) {
    
}

/* Matrix multiplication using shared memory
Input:  pointer to output data matrix, pointers to input matrix A and B,
        size of matrix A and width of Matrix B, and current GPUid
        Assumes width of A matches height of B; and shared memory view
Output: multiplication output saved in location pointed by output data matrix
*/
__global__
void FFNNFP_shared(data_t *out, data_t *A, data_t *B, int Ax, int Ay, int By) {

}

/* Matrix multiplication using Unified memory (from CUDA 6 onwards)
Input:  pointer to output data matrix, pointers to input matrix A and B,
        size of matrix A and width of Matrix B, and current GPUid
        Assumes width of A matches height of B; and unified memory view
Output: multiplication output saved in location pointed by output data matrix
*/
__global__
void FFNNFP_uni(data_t *out, data_t *A, data_t *B, int Ax, int Ay, int By) {

}
