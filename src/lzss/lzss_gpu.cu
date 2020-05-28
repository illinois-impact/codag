#include <common.h>
#include <lzss/lzss_gpu2.h>
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
	    lzss::compress_gpu(in_, &out_, in_sb.st_size, &out_size);
	}
	else {
	    lzss::decompress_gpu(in_, &out_, in_sb.st_size, &out_size);
	}
	std::chrono::high_resolution_clock::time_point compress_end = std::chrono::high_resolution_clock::now();

	fstat(out_fd, &out_sb);
	if (out_sb.st_size != out_size) {
	    if(ftruncate(out_fd, out_size) == -1) PRINT_ERROR;
	}
	out = mmap(nullptr, out_size, PROT_WRITE | PROT_READ, MAP_SHARED, out_fd, 0);

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
