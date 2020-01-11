#include <stdio.h>
#include <assert.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include <brle/brle.h>


void load_input(uint8_t *host_in, uint64_t *len) {
    for (int i=0; i<INPUT_SIZE; ++i) {
        host_in[i] = 'a' + rand()%26;
    }
    *len = INPUT_SIZE;
}

int main() {
    srand(time(NULL));
    const int len = sizeof(uint8_t) * INPUT_SIZE;

    uint8_t *host_in, *host_uncompress, *device_in, *device_out, *device_uncompress;
    uint64_t *device_ptr;
    host_in = (uint8_t *)malloc(len);
    host_uncompress = (uint8_t *)malloc(len);
    cudaMalloc((void**)&device_in, len);
    cudaMalloc((void**)&device_uncompress, len);

    const uint64_t num_threads = (INPUT_SIZE - 1) / CHUNK_SIZE + 1;
    const uint64_t num_pos = num_threads + 1; //0 at the front
    cudaMalloc((void**)&device_ptr, sizeof(uint64_t) * num_pos);
    cudaMalloc((void**)&device_ptr, sizeof(uint64_t) * num_pos);
    cudaMalloc((void**)&device_out, OUTPUT_CHUNK_SIZE * num_threads);

    uint64_t arr_len;
    load_input(host_in, &arr_len);
    cudaMemcpy(device_in, host_in, arr_len, cudaMemcpyHostToDevice);
    
    brle::kernel_compress<<<1, num_threads>>>(device_in, device_out, device_ptr, INPUT_SIZE);
    thrust::inclusive_scan(thrust::device, device_ptr, device_ptr + num_pos, device_ptr);
    brle::kernel_decompress<<<1, num_threads>>>(device_out, device_ptr, device_uncompress);

    cudaMemcpy(host_uncompress, device_uncompress, sizeof(uint8_t) * INPUT_SIZE, cudaMemcpyDeviceToHost);
  
    cudaDeviceSynchronize();

    for (uint64_t i=0; i<INPUT_SIZE; ++i) {
        assert(host_in[i] == host_uncompress[i]);
    }

    free(host_in);
    free(host_uncompress);
    cudaFree(device_uncompress);
    cudaFree(device_in);
    cudaFree(device_out);
    cudaFree(device_ptr);
    return 0;
}
