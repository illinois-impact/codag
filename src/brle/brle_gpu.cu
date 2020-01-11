// #include <stdio.h>
// #include <assert.h>
// #include <thrust/scan.h>
// #include <thrust/execution_policy.h>

// #include <brle/brle.h>


// void load_input(uint8_t *host_in, uint64_t *len) {
//     for (int i=0; i<INPUT_SIZE; ++i) {
//         host_in[i] = 'a' + rand()%26;
//     }
//     *len = INPUT_SIZE;
// }

// int main() {
//     srand(time(NULL));
//     const int len = sizeof(uint8_t) * INPUT_SIZE;

//     uint8_t *host_in, *host_uncompress, *device_in, *device_out, *device_uncompress;
//     uint64_t *device_ptr;
//     host_in = (uint8_t *)malloc(len);
//     host_uncompress = (uint8_t *)malloc(len);
//     cudaMalloc((void**)&device_in, len);
//     cudaMalloc((void**)&device_uncompress, len);

//     const uint64_t num_threads = (INPUT_SIZE - 1) / CHUNK_SIZE + 1;
//     const uint64_t num_pos = num_threads + 1; //0 at the front
//     cudaMalloc((void**)&device_ptr, sizeof(uint64_t) * num_pos);
//     cudaMalloc((void**)&device_ptr, sizeof(uint64_t) * num_pos);
//     cudaMalloc((void**)&device_out, OUTPUT_CHUNK_SIZE * num_threads);

//     uint64_t arr_len;
//     load_input(host_in, &arr_len);
//     cudaMemcpy(device_in, host_in, arr_len, cudaMemcpyHostToDevice);
    
//     brle::kernel_compress<<<1, num_threads>>>(device_in, device_out, device_ptr, INPUT_SIZE);
//     thrust::inclusive_scan(thrust::device, device_ptr, device_ptr + num_pos, device_ptr);
//     brle::kernel_decompress<<<1, num_threads>>>(device_out, device_ptr, device_uncompress);

//     cudaMemcpy(host_uncompress, device_uncompress, sizeof(uint8_t) * INPUT_SIZE, cudaMemcpyDeviceToHost);
  
//     cudaDeviceSynchronize();

//     for (uint64_t i=0; i<INPUT_SIZE; ++i) {
//         assert(host_in[i] == host_uncompress[i]);
//     }

//     free(host_in);
//     free(host_uncompress);
//     cudaFree(device_uncompress);
//     cudaFree(device_in);
//     cudaFree(device_out);
//     cudaFree(device_ptr);
//     return 0;
// }

#include <common.h>
#include <brle/brle.h>
#include <unistd.h>
#include <iostream>
#include <cstring>
#include <unistd.h>
#include <sys/types.h> 

#include <sys/stat.h> 
#include <fcntl.h>
#include <sys/mman.h>
#include <chrono>

int main(int argc, char** argv) {
    if (argc < 3) {
	std::cerr << "Please provide arguments\n";
	exit(1);
    }

    bool decomp = (strcmp(argv[1],"-d")==0);
    if (decomp && (argc < 4)) {
	std::cerr << "Please provide arguments\n";
	exit(1);
    }

    const char* input = decomp ? argv[2] : argv[1];
    const char* output = decomp ? argv[3] : argv[2];

    
	std::chrono::high_resolution_clock::time_point total_start = std::chrono::high_resolution_clock::now();
	int in_fd;
	struct stat in_sb;
	void* in;

	int out_fd;
	struct stat out_sb;
	void* out;

	if((in_fd = open(input, O_RDONLY)) == 0) {
	    printf("Fatal Error: INPUT File open error\n");
	    return -1;
	}
	if((out_fd = open(output, O_RDWR | O_TRUNC | O_CREAT, S_IRWXU | S_IRGRP | S_IROTH)) == 0) {
	    printf("Fatal Error: OUTPUT File open error\n");
	    return -1;
	}


	fstat(in_fd, &in_sb);

	in = mmap(nullptr, in_sb.st_size, PROT_READ, MAP_PRIVATE, in_fd, 0);

	if(in == (void*)-1){
	    printf("Fatal Error: INPUT Mapping error\n");
	    return -1;
	}
	uint8_t* in_ = (uint8_t*) in;
	uint8_t* out_;
	uint64_t out_size;
	std::chrono::high_resolution_clock::time_point compress_start = std::chrono::high_resolution_clock::now();
	if (!decomp) {
	    brle::compress_gpu(in_, &out_, in_sb.st_size, &out_size);
	}
	else {
	    // brle::decompress_gpu(in_, &out_, in_sb.st_size, &out_size);
    }

	std::chrono::high_resolution_clock::time_point compress_end = std::chrono::high_resolution_clock::now();

	fstat(out_fd, &out_sb);
	if (out_sb.st_size != out_size) {
	    if(ftruncate(out_fd, out_size) == -1) PRINT_ERROR;
	}
	out = mmap(nullptr, out_size, PROT_WRITE | PROT_READ, MAP_SHARED, out_fd, 0);

    printf("input size:%ld output bytes:%ld\n", in_sb.st_size, out_size);

    memcpy(out, out_, out_size);
    

	if(munmap(in, in_sb.st_size) == -1) PRINT_ERROR;
	if(munmap(out, out_size) == -1) PRINT_ERROR;

	close(in_fd);
	close(out_fd);
	std::chrono::high_resolution_clock::time_point total_end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> total = std::chrono::duration_cast<std::chrono::duration<double>>(total_end - total_start);
	std::chrono::duration<double> comp = std::chrono::duration_cast<std::chrono::duration<double>>(compress_end - compress_start);
	std::chrono::duration<double> wrt = std::chrono::duration_cast<std::chrono::duration<double>>(total_end - compress_end);
	std::cout << "Total time: " << total.count() << " secs\n";
	std::cout << "Compute time: " << comp.count() << " secs\n";
	std::cout << "Write time: " << wrt.count() << " secs\n";
    
}