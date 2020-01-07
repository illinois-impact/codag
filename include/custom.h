#ifndef CUSTOM_H
#define CUSTOM_H

#include <cuda.h>


struct CompressParams {
    size_t per_thread_in_block_size = 0;
    size_t per_thread_in_stride = 1;
    void (*compressor)(CompressParams*) = NULL;
    uint8_t* in = NULL;
    uint8_t* out = NULL;
    uint64_t* out_mapping = NULL;
    uint64_t n_blocks = 0;
    size_t in_size = 0;
    size_t out_size = 0;
    uint8_t compression_granularity = 1;
};

__global__ void compress(CompressParams* params) {
    uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    CompressParams p = *params;
    size_t in_start = tid*p.per_thread_in_block_size;
    if (in_start < p.in_size) {
       
	
    }

}


#endif
