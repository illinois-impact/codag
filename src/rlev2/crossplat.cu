#include <common.h>
#include <iostream>
#include <cstring>
#include <sys/types.h> 
#include <sys/stat.h> 
#include <fcntl.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include <chrono>

#include <rlev2/rlev2.h>
#include <cuda/atomic>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "please provide arguments\n";
        exit(1);
    }

    FILE* fp;
    long lSize;
    char* buffer;

    fp = fopen(argv[1], "rb");
    if (!fp) perror("fopen"), exit(EXIT_FAILURE);

    fseek(fp, 0L, SEEK_END);
    lSize = ftell(fp);
    rewind(fp);

    /* allocate memory for entire content */
    buffer = (char*)malloc(lSize);
    if (!buffer) fclose(fp), fputs("memory alloc fails", stderr), exit(EXIT_FAILURE);

    /* copy the file into the buffer */
    if (1 != fread(buffer, lSize, 1, fp))
        fclose(fp), free(buffer), fputs("entire read fails", stderr), exit(EXIT_FAILURE);

    uint8_t* encoded = nullptr;
    uint64_t encoded_bytes = 0;

    blk_off_t* blk_off;
    col_len_t* col_len;
    uint64_t n_chunks;

        
    INPUT_T* in = (INPUT_T*)buffer;

    uint8_t *t_out;
    auto encode_start = std::chrono::high_resolution_clock::now();

    // rlev2::compress_gpu(in, lSize, t_out, encoded_bytes);
    rlev2::compress_gpu_transpose<READ_GRANULARITY>(in, lSize, encoded, encoded_bytes, n_chunks, blk_off, col_len);

    auto encode_end = std::chrono::high_resolution_clock::now();

    INPUT_T* decoded = nullptr;
    uint64_t decoded_bytes = 0;

    auto decode_start = std::chrono::high_resolution_clock::now();

    // rlev2::decompress_gpu(t_out, encoded_bytes, decoded, decoded_bytes);
    rlev2::decompress_gpu<READ_GRANULARITY>(encoded, encoded_bytes, n_chunks, blk_off, col_len, decoded, decoded_bytes);
    
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

    printf("input size: %ld\n", lSize);   
    assert(decoded_bytes == lSize);
    printf("input ints: %ld\n", decoded_bytes / sizeof(INPUT_T));   

    for (int i = 0; i < decoded_bytes / sizeof(INPUT_T); ++i) {
        // printf("compare at %d %lld(%lld)\n", i, in[i], decoded[i]);
        if (decoded[i] != in[i]) {
            // printf("fail at %d %lld(%lld)\n", i, in[i], decoded[i]);

            for (int k=i; k<i+128; k+=32) {
            fprintf(stderr, "fail at %d %lld(%lld)\n", k, in[k], decoded[k]);

            }
        }
        assert(decoded[i] == in[i]);
    }
    
    /*
    for (int i = 0; i <128; ++i) {
        if (decoded[i] != in[i]) {
            printf("fail at %d %lld(%lld)\n", i, in[i], decoded[i]);
        }
    }
    */

    delete[] encoded;
    delete[] decoded;

    fclose(fp);
    free(buffer);

}
