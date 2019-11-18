/* Copyright 2019, Aman Gupta, ENG EC 527, Prof. Martin Herbordt
/******************************************************************************/
/* Neural Network Layer library for GPU in C
*/
#pragma once    /* file guard */
/******************************************************************************/
/* Libraries */
#include "matrix.cuh"

/******************************************************************************/
/* prototypes and usage */
/******************************************************************************/

/* Multi GPU prototypes */
/******************************************************************************/

/* Matrix multiplication using global memory
Input:  pointer to output data matrix, pointers to input matrix A and B,
        size of matrix A and width of Matrix B, and current GPUid
        Assumes width of A matches height of B; and global memory view
Output: multiplication output saved in location pointed by output data matrix
*/
__global__
void FFNNFP_global_mgpu(data_t *C, data_t *A, data_t *B, int Ax, int Ay, int By,
                        int GPUid);

/* Matrix multiplication using shared memory
Input:  pointer to output data matrix, pointers to input matrix A and B,
        size of matrix A and width of Matrix B, and current GPUid
        Assumes width of A matches height of B; and shared memory view
Output: multiplication output saved in location pointed by output data matrix
*/
__global__
void FFNNFP_shared_mgpu(data_t *C, data_t *A, data_t *B, int Ax, int Ay, int By,
                        int GPUid);

/* Matrix multiplication using Unified memory (from CUDA 6 onwards)
Input:  pointer to output data matrix, pointers to input matrix A and B,
        size of matrix A and width of Matrix B, and current GPUid
        Assumes width of A matches height of B; and unified memory view
Output: multiplication output saved in location pointed by output data matrix
*/
__global__
void FFNNFP_uni_mgpu(data_t *C, data_t *A, data_t *B, int Ax, int Ay, int By,
                    int GPUid);
