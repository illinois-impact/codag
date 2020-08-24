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

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "please provide arguments\n";
        exit(1);
    }

    int in_fd;
    struct stat in_sb;

    if((in_fd = open(argv[1], o_rdonly)) == 0) {
        printf("fatal error: input file open error\n");
        return -1;
    }
    fstat(in_fd, &in_sb);

    int64_t *in = (int64_t *)mmap(nullptr, in_sb.st_size, prot_read, map_private, in_fd, 0);
    if(in == (void*)-1){
        printf("fatal error: input mapping error\n");
        return -1;
    }
    close(in_fd);

    uint8_t *encoded = nullptr;
    uint64_t encoded_bytes = 0;

    blk_off_t *blk_off;
    col_len_t *col_len;
    uint64_t n_chunks;

    auto encode_start = std::chrono::high_resolution_clock::now();
    rlev2::compress_gpu_transpose(in, in_sb.st_size, encoded, encoded_bytes, n_chunks, blk_off, col_len);
    auto encode_end = std::chrono::high_resolution_clock::now();

    int64_t *decoded = nullptr;
    uint64_t decoded_bytes = 0;

    auto decode_start = std::chrono::high_resolution_clock::now();
    rlev2::decompress_gpu(encoded, encoded_bytes, n_chunks, blk_off, col_len, decoded, decoded_bytes);
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
    
    assert(decoded_bytes == in_sb.st_size);
    for (int i=0; i<decoded_bytes/sizeof(int64_t); ++i) {
         if (decoded[i] != in[i]) {
             printf("fail at %d %ld(%ld)\n", i, in[i], decoded[i]);
         }
        assert(decoded[i] == in[i]);
    }

    delete[] blk_off;
    delete[] col_len;
    delete[] encoded;
    delete[] decoded;
    if(munmap(in, in_sb.st_size) == -1) print_error;
}
