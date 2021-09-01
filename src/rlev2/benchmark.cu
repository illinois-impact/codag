#include <common.h>
#include <unistd.h>
#include <iostream>
#include <cstring>
#include <sys/types.h> 
#include <sys/stat.h> 
#include <fcntl.h>
#include <sys/mman.h>

#include <chrono>

#include <rlev2/rlev2.h>
#include <cuda/atomic>

template<int read_unit>
void benchmark(INPUT_T* in, uint64_t size) {
    uint8_t *encoded = nullptr;
    uint64_t encoded_bytes = 0;

    blk_off_t *blk_off;
    col_len_t *col_len;
    uint64_t n_chunks;

    auto encode_start = std::chrono::high_resolution_clock::now();
    rlev2::compress_gpu_transpose<read_unit>(in, size, encoded, encoded_bytes, n_chunks, blk_off, col_len);
    auto encode_end = std::chrono::high_resolution_clock::now();

    INPUT_T *decoded = nullptr;
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
    for (int i = 0; i < decoded_bytes / sizeof(INPUT_T); ++i) {
        if (decoded[i] != in[i]) {
            for (int k=i; k<i+read_unit; ++k) {
                fprintf(stderr, "fail at %d %u(%u)\n", k, in[k], decoded[k]);
            }
        }
        assert(decoded[i] == in[i]);
    }
    
    delete[] blk_off;
    delete[] col_len;
    delete[] encoded;
    delete[] decoded;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "please provide arguments\n";
        exit(1);
    }

    int in_fd;
    struct stat in_sb;

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

    benchmark<1>(in, in_sb.st_size);
    benchmark<4>(in, in_sb.st_size);
    benchmark<8>(in, in_sb.st_size);
    benchmark<16>(in, in_sb.st_size);
    benchmark<32>(in, in_sb.st_size);
    benchmark<64>(in, in_sb.st_size);
    benchmark<128>(in, in_sb.st_size);
    benchmark<256>(in, in_sb.st_size);


    
    if(munmap(in, in_sb.st_size) == -1) PRINT_ERROR;
}
