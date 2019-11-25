/******************************************************************************/
/* Parameters */
#define TILE_WIDTH              32
#define NUM_THREADS_PER_BLOCK   1024

/*
export PATH=/Developer/NVIDIA/CUDA-10.2/bin${PATH:+:${PATH}}
export DYLD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-10.2/lib\
                         ${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}
*/

CUDA_SAFE_CALL(cudaSetDevice(0));