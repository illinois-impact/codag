#include <common.h>
#include <brle/brle.h>
#include <unistd.h>
#include <iostream>
#include <cstring>
#include <sys/types.h> 

#include <sys/stat.h> 
#include <fcntl.h>
#include <sys/mman.h>
#include <chrono>

#include <cassert>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Please provide arguments\n";
        exit(1);
    }

    const char* filename = argv[1];

	int in_fd;
	struct stat in_sb;
	void* in;

    if((in_fd = open(filename, O_RDONLY)) == 0) {
	    printf("Fatal Error: INPUT File open error\n");
	    return -1;
	}


	fstat(in_fd, &in_sb);
	in = mmap(nullptr, in_sb.st_size, PROT_READ, MAP_PRIVATE, in_fd, 0);

	if(in == (void*)-1){
	    printf("Fatal Error: INPUT Mapping error\n");
	    return -1;
	}
	uint8_t* in_ = (uint8_t*) in;
	uint8_t* encoded, *decoded;
	uint64_t out_size;
    
    brle::compress_gpu(in_, &encoded, in_sb.st_size, &out_size);
    brle::decompress_gpu(encoded, &decoded, out_size, &out_size);
    
    assert(out_size == in_sb.st_size);
    for (uint64_t i=0; i<in_sb.st_size; ++i) {
        assert(in_[i] == decoded[i]);
    }

	if(munmap(in, in_sb.st_size) == -1) PRINT_ERROR;

	close(in_fd);
}