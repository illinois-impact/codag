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
        std::string metaname(output);
        struct stat meta_sb;
        metaname += ".meta";
        int meta_fd;
        if((meta_fd = open(metaname.c_str(), O_RDWR | O_TRUNC | O_CREAT, S_IRWXU | S_IRGRP | S_IROTH)) == 0) {
            printf("Fatal Error: OUTPUT Metadata File open error\n");
            return -1;
        }

        fstat(meta_fd, &meta_sb);
        uint64_t metadata_size = meta_sb.st_size;
        uint8_t** out_metadata;
        uint64_t* out_metadata_lens;
        uint64_t out_metadata_num;

	    lzss::compress_gpu(in_, in_sb.st_size, &out_, &out_size, &out_metadata, &out_metadata_lens, &out_metadata_num);

        //printf("Finished compression\n");
        metadata_size = sizeof(uint64_t) + sizeof(uint64_t) * out_metadata_num;
        for (size_t i = 0; i < out_metadata_num; i++) {
            metadata_size += out_metadata_lens[i];

        }


        if (meta_sb.st_size != metadata_size) {
            if(ftruncate(meta_fd, metadata_size) == -1) PRINT_ERROR;
        }
        uint8_t* out_metadata_f = (uint8_t*) mmap(nullptr, metadata_size, PROT_WRITE | PROT_READ, MAP_SHARED, meta_fd, 0);

        uint64_t offset = 0;
        memcpy(out_metadata_f+offset, &out_metadata_num, sizeof(uint64_t));
        offset += sizeof(uint64_t);

        memcpy(out_metadata_f+offset, out_metadata_lens, sizeof(uint64_t) * out_metadata_num);
        offset += sizeof(uint64_t) * out_metadata_num;

        for (size_t i = 0; i < out_metadata_num; i++) {
            memcpy(out_metadata_f+offset, out_metadata[i], out_metadata_lens[i]);
            offset += out_metadata_lens[i];
        }
        if(munmap(out_metadata_f, metadata_size) == -1) PRINT_ERROR;
        close(meta_fd);

	}
	else {
        std::string metaname(input);
        struct stat meta_sb;
        metaname += ".meta";
        int meta_fd;
        if((meta_fd = open(metaname.c_str(),  O_RDONLY)) == 0) {
            printf("Fatal Error: OUTPUT Metadata File open error\n");
            return -1;
        }

        printf("file: %s\n", metaname.c_str());

        fstat(meta_fd, &meta_sb);
        uint64_t metadata_size = meta_sb.st_size;
        int ret;
        uint8_t* in_metadata_f = (uint8_t*) mmap(nullptr, metadata_size, PROT_READ, MAP_PRIVATE, meta_fd, 0);
        uint64_t metadata_num = ((uint64_t*) in_metadata_f)[0];
        uint64_t* metadata_lens = (uint64_t*) (in_metadata_f + sizeof(uint64_t));

        uint8_t** metadata = (uint8_t**) malloc(metadata_num * sizeof(uint8_t*));
        uint64_t offset = sizeof(uint64_t) + sizeof(uint64_t) * metadata_num;
        for (size_t i = 0; i < metadata_num; i++) {
            printf("offset: %llu\n", offset);
            metadata[i] = (uint8_t*) (in_metadata_f + offset);
            offset += metadata_lens[i];
        }


	    ret = lzss::decompress_gpu(in_, in_sb.st_size, (const uint8_t**) metadata, metadata_lens, metadata_num, &out_, &out_size);

        if(munmap(in_metadata_f, metadata_size) == -1) PRINT_ERROR;
        close(meta_fd);
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
