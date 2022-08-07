#include <common.h>
#include <unistd.h>
#include <iostream>
#include <cstring>
#include <sys/types.h> 
#include <sys/stat.h> 
#include <fcntl.h>
#include <sys/mman.h>

#include <chrono>

#include <fstream>


#include <rlev2/rlev2.h>
//#include <cuda/atomic>

void check_diff(INPUT_T* in, INPUT_T* decoded, uint64_t size, uint64_t decoded_bytes, int read_unit){
   assert(decoded_bytes == size);


   uint8_t* in2 = (uint8_t*) in;
   uint8_t* decoded2 = (uint8_t*) decoded;
   for(uint64_t i = 0; i < decoded_bytes; i++){
        if (decoded2[i] != in2[i]) {
            printf( "fail at %llu %lx(%lx)\n", i, in2[i], decoded2[i]);
            break;
        }
   }


 //   for (int i = 0; i < decoded_bytes / sizeof(INPUT_T); ++i) {
	// if (decoded[i] != in[i]) {
	// 	for (int k=i; k<i+read_unit; ++k) {
	// 		printf( "fail at %d %u(%u)\n", k, in[k], decoded[k]);
	//         }
	//         break;
 //          }
 //    }
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
    rlev2::compress_gpu_orig<read_unit, COMP_TYPE>(in, size, encoded, encoded_bytes, meta_data_bytes, n_chunks, blk_off, col_len, CHUNK_SIZE);
    auto encode_end = std::chrono::high_resolution_clock::now();

    INPUT_T *decoded = nullptr;
    uint64_t decoded_bytes = 0;

    // std::cout << CHUNK_SIZE << "\t" << (read_unit * COL_WIDTH) << " \t" << 2 << "\t" << encoded_bytes << "\t" << meta_data_bytes << "\t";
    // rlev2::decompress_gpu<read_unit,2, COMP_TYPE>(encoded, encoded_bytes, n_chunks, blk_off, col_len, decoded, decoded_bytes, CHUNK_SIZE); 
    // check_diff(in, decoded, size, decoded_bytes, read_unit); 

    // std::cout << CHUNK_SIZE << "\t" << (read_unit * COL_WIDTH) << " \t" << 4 << "\t" << encoded_bytes << "\t" << meta_data_bytes << "\t";
    // rlev2::decompress_gpu<read_unit,4, COMP_TYPE>(encoded, encoded_bytes, n_chunks, blk_off, col_len, decoded, decoded_bytes, CHUNK_SIZE);
    // check_diff(in, decoded, size, decoded_bytes, read_unit);    
   
    // std::cout << CHUNK_SIZE << "\t" << (read_unit * COL_WIDTH) << " \t" << 8 << "\t" << encoded_bytes << "\t" << meta_data_bytes << "\t";
    // rlev2::decompress_gpu<read_unit,8, COMP_TYPE>(encoded, encoded_bytes, n_chunks, blk_off, col_len, decoded, decoded_bytes, CHUNK_SIZE);
    // check_diff(in, decoded, size, decoded_bytes, read_unit);

    // std::cout << CHUNK_SIZE << "\t" << (read_unit * COL_WIDTH) << " \t" << 16 << "\t" << encoded_bytes << "\t" << meta_data_bytes << "\t";
    // rlev2::decompress_gpu<read_unit,16, COMP_TYPE>(encoded, encoded_bytes, n_chunks, blk_off, col_len, decoded, decoded_bytes, CHUNK_SIZE);
    // check_diff(in, decoded, size, decoded_bytes, read_unit);

    // std::cout << CHUNK_SIZE << "\t" << (read_unit * COL_WIDTH) << " \t" << 24 << "\t" << encoded_bytes << "\t" << meta_data_bytes << "\t";
    // rlev2::decompress_gpu<read_unit,24, COMP_TYPE>(encoded, encoded_bytes, n_chunks, blk_off, col_len, decoded, decoded_bytes, CHUNK_SIZE);
    // check_diff(in, decoded, size, decoded_bytes, read_unit);

    // std::cout << CHUNK_SIZE << "\t" << (read_unit * COL_WIDTH) << " \t" << 32 << "\t" << encoded_bytes << "\t" << meta_data_bytes << "\t";
    // rlev2::decompress_gpu<read_unit,32, COMP_TYPE>(encoded, encoded_bytes, n_chunks, blk_off, col_len, decoded, decoded_bytes, CHUNK_SIZE);
    // check_diff(in, decoded, size, decoded_bytes, read_unit);

    std::cout << CHUNK_SIZE  << " \t" << 48 << "\t" << encoded_bytes << "\t" << meta_data_bytes << "\t";
    //rlev2::decompress_gpu<read_unit,48, COMP_TYPE>(encoded, encoded_bytes, n_chunks, blk_off, col_len, decoded, decoded_bytes, CHUNK_SIZE);

    rlev2::decompress_gpu_orig<read_unit,48, COMP_TYPE, 2, 1>(encoded, encoded_bytes, n_chunks, blk_off, col_len, decoded, decoded_bytes, CHUNK_SIZE);
     check_diff(in, decoded, size, decoded_bytes, read_unit);
    rlev2::decompress_gpu_orig<read_unit,48, COMP_TYPE, 2, 2>(encoded, encoded_bytes, n_chunks, blk_off, col_len, decoded, decoded_bytes, CHUNK_SIZE);
    check_diff(in, decoded, size, decoded_bytes, read_unit);
    rlev2::decompress_gpu_orig<read_unit,48, COMP_TYPE, 2, 4>(encoded, encoded_bytes, n_chunks, blk_off, col_len, decoded, decoded_bytes, CHUNK_SIZE);
    check_diff(in, decoded, size, decoded_bytes, read_unit);
    rlev2::decompress_gpu_orig<read_unit,48, COMP_TYPE, 2, 8>(encoded, encoded_bytes, n_chunks, blk_off, col_len, decoded, decoded_bytes, CHUNK_SIZE);
    check_diff(in, decoded, size, decoded_bytes, read_unit);
    rlev2::decompress_gpu_orig<read_unit,48, COMP_TYPE, 2, 16>(encoded, encoded_bytes, n_chunks, blk_off, col_len, decoded, decoded_bytes, CHUNK_SIZE);
    check_diff(in, decoded, size, decoded_bytes, read_unit);


    std::cout << CHUNK_SIZE  << " \t" << 48 << "\t" << encoded_bytes << "\t" << meta_data_bytes << "\t";
    rlev2::decompress_gpu_orig<read_unit,64, COMP_TYPE, 1, 1>(encoded, encoded_bytes, n_chunks, blk_off, col_len, decoded, decoded_bytes, CHUNK_SIZE);
    check_diff(in, decoded, size, decoded_bytes, read_unit);
    rlev2::decompress_gpu_orig<read_unit,64, COMP_TYPE, 1, 2>(encoded, encoded_bytes, n_chunks, blk_off, col_len, decoded, decoded_bytes, CHUNK_SIZE);
    check_diff(in, decoded, size, decoded_bytes, read_unit);
    rlev2::decompress_gpu_orig<read_unit,64, COMP_TYPE, 1, 4>(encoded, encoded_bytes, n_chunks, blk_off, col_len, decoded, decoded_bytes, CHUNK_SIZE);
    check_diff(in, decoded, size, decoded_bytes, read_unit);
    rlev2::decompress_gpu_orig<read_unit,64, COMP_TYPE, 1, 8>(encoded, encoded_bytes, n_chunks, blk_off, col_len, decoded, decoded_bytes, CHUNK_SIZE);
    check_diff(in, decoded, size, decoded_bytes, read_unit);
    rlev2::decompress_gpu_orig<read_unit,64, COMP_TYPE, 1, 16>(encoded, encoded_bytes, n_chunks, blk_off, col_len, decoded, decoded_bytes, CHUNK_SIZE);
    check_diff(in, decoded, size, decoded_bytes, read_unit);

    delete[] blk_off;
    delete[] col_len;
    delete[] encoded;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "please provide arguments INPUT_FILE SIZE_OF_DATATYPE\n";
        exit(1);
    }

    int in_fd;
    struct stat in_sb;

    int datatype_size = atoi(argv[2]);
    if((in_fd = open(argv[1], O_RDONLY)) == 0) {
        printf("fatal error: input file open error\n");
        return -1;
    }
    fstat(in_fd, &in_sb);

    INPUT_T *in = (INPUT_T *)mmap(nullptr, in_sb.st_size, PROT_READ, MAP_PRIVATE, in_fd, 0);
    if(in == (void*)-1){
        printf("fatal error: input mapping error\n");
        return -1;
    }
    close(in_fd);

  // benchmark<1>(in, in_sb.st_size);
 //   benchmark<4>(in, in_sb.st_size);
    uint64_t CHUNK_SIZE = 1024 * 128;
    int CW_ITR = 1;
    // CW_ITR = 6;
    std::cout << "CHUNK_SIZE" << "\t" << " BUFFER SIZE \t COMPRESSED SIZE \t META DATA SIZE \t DECOMPRESSION TIME\n";


    if(datatype_size == 1){
    	for(int c = 0; c < CW_ITR; c++){
   // 	 	benchmark<8,uint8_t>(in, in_sb.st_size, CHUNK_SIZE);
   // 	 	benchmark<16,uint8_t>(in, in_sb.st_size, CHUNK_SIZE); 
    	 	benchmark<32,uint8_t>(in, in_sb.st_size, CHUNK_SIZE);
	 	// benchmark<64,uint8_t>(in, in_sb.st_size, CHUNK_SIZE);
	 	// benchmark<128,uint8_t>(in, in_sb.st_size, CHUNK_SIZE);
   // 	 	benchmark<256,uint8_t>(in, in_sb.st_size, CHUNK_SIZE);
		CHUNK_SIZE += (1024 * 8);
    	}
    }
    else if(datatype_size == 2){
    	for(int c = 0; c < CW_ITR; c++){
  //  	 	benchmark<4,uint16_t>(in, in_sb.st_size, CHUNK_SIZE);
		// benchmark<8,uint16_t>(in, in_sb.st_size, CHUNK_SIZE);
    	 	benchmark<16,uint16_t>(in, in_sb.st_size, CHUNK_SIZE); 
  //  	 	benchmark<32,uint16_t>(in, in_sb.st_size, CHUNK_SIZE);
	 // 	benchmark<64,uint16_t>(in, in_sb.st_size, CHUNK_SIZE);
	 // 	benchmark<128,uint16_t>(in, in_sb.st_size, CHUNK_SIZE);
		CHUNK_SIZE += (1024 * 8);
    	}
    }
    else if(datatype_size == 4){
    	for(int c = 0; c < CW_ITR; c++){
  //  	        benchmark<2,uint32_t>(in, in_sb.st_size, CHUNK_SIZE); 
		// benchmark<4,uint32_t>(in, in_sb.st_size, CHUNK_SIZE);
		 benchmark<8,uint32_t>(in, in_sb.st_size, CHUNK_SIZE);
  //  	 	benchmark<16,uint32_t>(in, in_sb.st_size, CHUNK_SIZE); 
  //  	 	benchmark<32,uint32_t>(in, in_sb.st_size, CHUNK_SIZE);
	 // 	benchmark<64,uint32_t>(in, in_sb.st_size, CHUNK_SIZE);
		CHUNK_SIZE += (1024 * 8);
    	}
    }

    if(datatype_size == 8){
    	for(int c = 0; c < CW_ITR; c++){
   	    //benchmark<1,uint64_t>(in, in_sb.st_size, CHUNK_SIZE);
	    //benchmark<2,uint64_t>(in, in_sb.st_size, CHUNK_SIZE);
		benchmark<4,uint64_t>(in, in_sb.st_size, CHUNK_SIZE);
		//benchmark<8,uint64_t>(in, in_sb.st_size, CHUNK_SIZE);
   	 	//benchmark<16,uint64_t>(in, in_sb.st_size, CHUNK_SIZE); 
   	 	//benchmark<32,uint64_t>(in, in_sb.st_size, CHUNK_SIZE);
		CHUNK_SIZE += (1024 * 8);
    	}
    }




    if(munmap(in, in_sb.st_size) == -1) PRINT_ERROR;
}

