#include <common.h>
#include <unistd.h>
#include <iostream>
#include <cstring>
#include <sys/types.h> 
#include <sys/stat.h> 
#include <fcntl.h>
#include <sys/mman.h>

#include <chrono>

<<<<<<< HEAD
#include <rlev2/rlev2.h>
#include <cuda/atomic>

template<int read_unit>
void benchmark(int64_t* in, uint64_t size) {
    uint8_t *encoded = nullptr;
    uint64_t encoded_bytes = 0;

    blk_off_t *blk_off;
    col_len_t *col_len;
    uint64_t n_chunks;

    auto encode_start = std::chrono::high_resolution_clock::now();
    rlev2::compress_gpu_transpose<read_unit>(in, size, encoded, encoded_bytes, n_chunks, blk_off, col_len);
    auto encode_end = std::chrono::high_resolution_clock::now();

    int64_t *decoded = nullptr;
    uint64_t decoded_bytes = 0;

    auto decode_start = std::chrono::high_resolution_clock::now();
    rlev2::decompress_gpu<read_unit>(encoded, encoded_bytes, n_chunks, blk_off, col_len, decoded, decoded_bytes);
    auto decode_end = std::chrono::high_resolution_clock::now();
        
    auto decomp = std::chrono::duration_cast<std::chrono::duration<double>>(decode_end - decode_start);
    std::cout << "decompression size: " << encoded_bytes << " bytes\n";
    std::cout << "decompression time: " << decomp.count() << " secs\n";
    // printf("exp(actual) %lu(%lu)\n",decoded_bytes, sizeof(ll));
    // for (int i=0; i<n_digits; ++i) {
    //     if (ll[i] != decoded[i]) {
    //         printf("failed at %d\n", i);
    //         break;

    //     }
    //     // printf("%ld : %ld\n", ll[i], decompressed[i]);
    // }
    
    assert(decoded_bytes == size);
    for (int i=0; i<decoded_bytes/sizeof(int64_t); ++i) {
        if (decoded[i] != in[i]) {
            printf("fail at %d %ld(%ld)\n", i, in[i], decoded[i]);
        }
        assert(decoded[i] == in[i]);
    }
=======
#include <fstream>


#include <rlev2/rlev2.h>
#include <cuda/atomic>

void check_diff(INPUT_T* in, INPUT_T* decoded, uint64_t size, uint64_t decoded_bytes, int read_unit){
   assert(decoded_bytes == size);
   for (int i = 0; i < decoded_bytes / sizeof(INPUT_T); ++i) {
	if (decoded[i] != in[i]) {
		for (int k=i; k<i+read_unit; ++k) {
			printf( "fail at %d %u(%u)\n", k, in[k], decoded[k]);
	        }
	        break;
          }
    }
    delete[] decoded;


}
template<int read_unit,  typename COMP_TYPE>
void benchmark(INPUT_T* in, uint64_t size, uint64_t CHUNK_SIZE) {
    uint8_t *encoded = nullptr;
    uint64_t encoded_bytes = 0;

    uint64_t meta_data_bytes = 0;
    blk_off_t *blk_off;
    col_len_t *col_len;
    uint64_t n_chunks;
    int COL_WIDTH = sizeof(COMP_TYPE);
    auto encode_start = std::chrono::high_resolution_clock::now();
    rlev2::compress_gpu_transpose<read_unit, COMP_TYPE>(in, size, encoded, encoded_bytes, meta_data_bytes, n_chunks, blk_off, col_len, CHUNK_SIZE);
    auto encode_end = std::chrono::high_resolution_clock::now();

    INPUT_T *decoded = nullptr;
    uint64_t decoded_bytes = 0;

    std::cout << CHUNK_SIZE << "\t" << (read_unit * COL_WIDTH) << " \t" << 2 << "\t" << encoded_bytes << "\t" << meta_data_bytes << "\t";
    rlev2::decompress_gpu<read_unit,2, COMP_TYPE>(encoded, encoded_bytes, n_chunks, blk_off, col_len, decoded, decoded_bytes, CHUNK_SIZE); 
    check_diff(in, decoded, size, decoded_bytes, read_unit); 

    std::cout << CHUNK_SIZE << "\t" << (read_unit * COL_WIDTH) << " \t" << 4 << "\t" << encoded_bytes << "\t" << meta_data_bytes << "\t";
    rlev2::decompress_gpu<read_unit,4, COMP_TYPE>(encoded, encoded_bytes, n_chunks, blk_off, col_len, decoded, decoded_bytes, CHUNK_SIZE);
    check_diff(in, decoded, size, decoded_bytes, read_unit);    
   
    std::cout << CHUNK_SIZE << "\t" << (read_unit * COL_WIDTH) << " \t" << 8 << "\t" << encoded_bytes << "\t" << meta_data_bytes << "\t";
    rlev2::decompress_gpu<read_unit,8, COMP_TYPE>(encoded, encoded_bytes, n_chunks, blk_off, col_len, decoded, decoded_bytes, CHUNK_SIZE);
    check_diff(in, decoded, size, decoded_bytes, read_unit);

    std::cout << CHUNK_SIZE << "\t" << (read_unit * COL_WIDTH) << " \t" << 16 << "\t" << encoded_bytes << "\t" << meta_data_bytes << "\t";
    rlev2::decompress_gpu<read_unit,16, COMP_TYPE>(encoded, encoded_bytes, n_chunks, blk_off, col_len, decoded, decoded_bytes, CHUNK_SIZE);
    check_diff(in, decoded, size, decoded_bytes, read_unit);

    std::cout << CHUNK_SIZE << "\t" << (read_unit * COL_WIDTH) << " \t" << 24 << "\t" << encoded_bytes << "\t" << meta_data_bytes << "\t";
    rlev2::decompress_gpu<read_unit,24, COMP_TYPE>(encoded, encoded_bytes, n_chunks, blk_off, col_len, decoded, decoded_bytes, CHUNK_SIZE);
    check_diff(in, decoded, size, decoded_bytes, read_unit);

    std::cout << CHUNK_SIZE << "\t" << (read_unit * COL_WIDTH) << " \t" << 32 << "\t" << encoded_bytes << "\t" << meta_data_bytes << "\t";
    rlev2::decompress_gpu<read_unit,32, COMP_TYPE>(encoded, encoded_bytes, n_chunks, blk_off, col_len, decoded, decoded_bytes, CHUNK_SIZE);
    check_diff(in, decoded, size, decoded_bytes, read_unit);

    std::cout << CHUNK_SIZE << "\t" << (read_unit * COL_WIDTH) << " \t" << 48 << "\t" << encoded_bytes << "\t" << meta_data_bytes << "\t";
    rlev2::decompress_gpu<read_unit,48, COMP_TYPE>(encoded, encoded_bytes, n_chunks, blk_off, col_len, decoded, decoded_bytes, CHUNK_SIZE);
    check_diff(in, decoded, size, decoded_bytes, read_unit);

>>>>>>> 2521a77ef53923581b24ccb056259d35bc9e5a2d

    delete[] blk_off;
    delete[] col_len;
    delete[] encoded;
<<<<<<< HEAD
    delete[] decoded;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "please provide arguments\n";
=======
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "please provide arguments INPUT_FILE SIZE_OF_DATATYPE\n";
>>>>>>> 2521a77ef53923581b24ccb056259d35bc9e5a2d
        exit(1);
    }

    int in_fd;
    struct stat in_sb;

<<<<<<< HEAD
=======
    int datatype_size = atoi(argv[2]);
>>>>>>> 2521a77ef53923581b24ccb056259d35bc9e5a2d
    if((in_fd = open(argv[1], O_RDONLY)) == 0) {
        printf("fatal error: input file open error\n");
        return -1;
    }
    fstat(in_fd, &in_sb);

<<<<<<< HEAD
    int64_t *in = (int64_t *)mmap(nullptr, in_sb.st_size, PROT_READ, MAP_PRIVATE, in_fd, 0);
=======
    INPUT_T *in = (INPUT_T *)mmap(nullptr, in_sb.st_size, PROT_READ, MAP_PRIVATE, in_fd, 0);
>>>>>>> 2521a77ef53923581b24ccb056259d35bc9e5a2d
    if(in == (void*)-1){
        printf("fatal error: input mapping error\n");
        return -1;
    }
    close(in_fd);

<<<<<<< HEAD
    benchmark<1>(in, in_sb.st_size);
//    benchmark<2>(in, in_sb.st_size);
//    benchmark<4>(in, in_sb.st_size);
//    benchmark<8>(in, in_sb.st_size);

    
=======
  // benchmark<1>(in, in_sb.st_size);
 //   benchmark<4>(in, in_sb.st_size);
    uint64_t CHUNK_SIZE = 1024 * 8;
    std::cout << "CHUNK_SIZE" << "\t" << "COLUMN WIDTH\t BUFFER SIZE \t COMPRESSED SIZE \t META DATA SIZE \t DECOMPRESSION TIME\n";

    if(datatype_size == 1){
    	for(int c = 0; c < 6; c++){
   	 	benchmark<8,uint8_t>(in, in_sb.st_size, CHUNK_SIZE);
   	 	benchmark<16,uint8_t>(in, in_sb.st_size, CHUNK_SIZE); 
   	 	benchmark<32,uint8_t>(in, in_sb.st_size, CHUNK_SIZE);
	 	benchmark<64,uint8_t>(in, in_sb.st_size, CHUNK_SIZE);
	 	benchmark<128,uint8_t>(in, in_sb.st_size, CHUNK_SIZE);
   	 	benchmark<256,uint8_t>(in, in_sb.st_size, CHUNK_SIZE);
		CHUNK_SIZE += (1024 * 8);
    	}
    }
    else if(datatype_size == 2){
    	for(int c = 0; c < 6; c++){
   	 	benchmark<4,uint16_t>(in, in_sb.st_size, CHUNK_SIZE);
		benchmark<8,uint16_t>(in, in_sb.st_size, CHUNK_SIZE);
   	 	benchmark<16,uint16_t>(in, in_sb.st_size, CHUNK_SIZE); 
   	 	benchmark<32,uint16_t>(in, in_sb.st_size, CHUNK_SIZE);
	 	benchmark<64,uint16_t>(in, in_sb.st_size, CHUNK_SIZE);
	 	benchmark<128,uint16_t>(in, in_sb.st_size, CHUNK_SIZE);
		CHUNK_SIZE += (1024 * 8);
    	}
    }
    else if(datatype_size == 4){
    	for(int c = 0; c < 6; c++){
   	        benchmark<2,uint32_t>(in, in_sb.st_size, CHUNK_SIZE); 
		benchmark<4,uint32_t>(in, in_sb.st_size, CHUNK_SIZE);
		benchmark<8,uint32_t>(in, in_sb.st_size, CHUNK_SIZE);
   	 	benchmark<16,uint32_t>(in, in_sb.st_size, CHUNK_SIZE); 
   	 	benchmark<32,uint32_t>(in, in_sb.st_size, CHUNK_SIZE);
	 	benchmark<64,uint32_t>(in, in_sb.st_size, CHUNK_SIZE);
		CHUNK_SIZE += (1024 * 8);
    	}
    }

    else if(datatype_size == 8){
    	for(int c = 0; c < 6; c++){
   	        benchmark<1,uint64_t>(in, in_sb.st_size, CHUNK_SIZE);
	        benchmark<2,uint64_t>(in, in_sb.st_size, CHUNK_SIZE);
		benchmark<4,uint64_t>(in, in_sb.st_size, CHUNK_SIZE);
		benchmark<8,uint64_t>(in, in_sb.st_size, CHUNK_SIZE);
   	 	benchmark<16,uint64_t>(in, in_sb.st_size, CHUNK_SIZE); 
   	 	benchmark<32,uint64_t>(in, in_sb.st_size, CHUNK_SIZE);
		CHUNK_SIZE += (1024 * 8);
    	}
    }




>>>>>>> 2521a77ef53923581b24ccb056259d35bc9e5a2d
    if(munmap(in, in_sb.st_size) == -1) PRINT_ERROR;
}
