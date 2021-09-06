// #ifndef __ZLIB_H__
// #define __ZLIB_H__

#include <common.h>
#include <fstream>
#include <sys/types.h> 
#include <sys/stat.h> 
#include <fcntl.h>
#include <sys/mman.h>
#include <simt/atomic>
#include <iostream>

#define BUFF_LEN 2




#include "encoder.h"
#include "decoder.h"


namespace rle_v1 {

template <typename READ_COL_TYPE, typename DATA_TYPE, typename OUT_COL_TYPE>
 __host__ void compress_gpu(const uint8_t* const in, uint8_t** out, const uint64_t in_n_bytes, uint64_t* out_n_bytes, int COMP_COL_LEN, uint64_t CHUNK_SIZE){
// uint64_t* col_len_f,  uint64_t* blk_offset_f, uint64_t chunk_size) {
  //  uint64_t CHUNK_SIZE = 1024 * 8;

    uint32_t num_blk = ((in_n_bytes + CHUNK_SIZE - 1) / CHUNK_SIZE);

    const uint64_t col_n_bytes = sizeof(uint64_t) * 32 * (num_blk);
    const uint64_t blk_n_bytes = sizeof(uint64_t) * (num_blk + 1);
    uint8_t* d_in;
    uint64_t* d_col_len;
    uint64_t* d_blk_offset;
    uint8_t* d_out;

    cuda_err_chk(cudaMalloc(&d_in, num_blk * CHUNK_SIZE));
    cuda_err_chk(cudaMalloc(&d_col_len,col_n_bytes));
    cuda_err_chk(cudaMalloc(&d_blk_offset, blk_n_bytes));

    cuda_err_chk(cudaMemset(d_in, 0, num_blk * CHUNK_SIZE));

    cuda_err_chk(cudaMemcpy(d_in, in, in_n_bytes, cudaMemcpyHostToDevice));

    dim3 blockD(32,2,1);
    dim3 gridD(num_blk,1,1);
    cuda_err_chk(cudaDeviceSynchronize());

    std::chrono::high_resolution_clock::time_point kernel_start = std::chrono::high_resolution_clock::now();
    
    setup_deflate<READ_COL_TYPE, DATA_TYPE, uint32_t, 8> <<<gridD,blockD>>>   (d_in, d_col_len, d_blk_offset, CHUNK_SIZE, COMP_COL_LEN);

    cuda_err_chk(cudaDeviceSynchronize());
    std::chrono::high_resolution_clock::time_point kernel_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total = std::chrono::duration_cast<std::chrono::duration<double>>(kernel_end - kernel_start);

    reduction_scan<<<1,1>>>(d_blk_offset, num_blk);
    cuda_err_chk(cudaDeviceSynchronize());

    uint64_t* h_col_len = new uint64_t[32 * num_blk];
    cuda_err_chk(cudaMemcpy(h_col_len, d_col_len, col_n_bytes, cudaMemcpyDeviceToHost));

    uint64_t* h_blk_off = new uint64_t[num_blk+1];
    cuda_err_chk(cudaMemcpy(h_blk_off, d_blk_offset, blk_n_bytes, cudaMemcpyDeviceToHost));

    *out_n_bytes = h_blk_off[num_blk]+8;
    *out = new uint8_t[*out_n_bytes];

    ((uint64_t*)(*out))[0] = in_n_bytes;

    cuda_err_chk(cudaMalloc(&d_out, (*out_n_bytes)));

    dim3 blockD_comp(32,2,1);

    deflate<READ_COL_TYPE, DATA_TYPE, uint32_t, 8> <<<gridD,blockD_comp>>>   (d_in, d_out, d_col_len, d_blk_offset, CHUNK_SIZE, COMP_COL_LEN);
    cuda_err_chk(cudaDeviceSynchronize());

    cuda_err_chk(cudaMemcpy(&((*out)[8]), d_out, *out_n_bytes, cudaMemcpyDeviceToHost));

    std::ofstream col_len_file("./input_data/rle_v1_col_len.bin", std::ofstream::binary);
    col_len_file.write((const char *)(h_col_len), 32 * num_blk * 8);
    col_len_file.close();

    std::ofstream blk_off_file("./input_data/rle_v1_blk_offset.bin", std::ofstream::binary);
    blk_off_file.write((const char *)(h_blk_off), (num_blk + 1) * 8);
    blk_off_file.close();




    cudaFree(d_in);
    cudaFree(d_col_len);
    cudaFree(d_blk_offset);
    cudaFree(d_out);

}

template <typename READ_COL_TYPE, typename DATA_TYPE, typename OUT_COL_TYPE, int QUEUE_SIZE = 8>
__host__ void decompress_gpu(const uint8_t* const in, uint8_t** out, const uint64_t in_n_bytes, uint64_t* out_n_bytes, int COMP_COL_LEN, uint64_t chunk_size){
  // uint64_t* col_len_f, const uint64_t col_n_bytes, uint64_t* blk_offset_f, const uint64_t blk_n_bytes) {



    cudaSetDevice(1);
    uint8_t* d_in;
    uint64_t* d_col_len;
    uint64_t* d_blk_offset;

    std::string file_col_len = "./input_data/rle_v1_col_len.bin";
    std::string file_blk_off = "./input_data/rle_v1_blk_offset.bin";

    const char *filename_col_len = file_col_len.c_str();
    const char *filename_blk_off = file_blk_off.c_str();

    int fd_col_len;
    int fd_blk_off;

    struct stat sbcol_len;
    struct stat sbblk_off;

    if ((fd_col_len = open(filename_col_len, O_RDONLY)) == -1) {
        printf("Fatal Error: Col Len read error\n");
        return;
    }
    if ((fd_blk_off = open(filename_blk_off, O_RDONLY)) == -1) {
        printf("Fatal Error: Block off read error\n");
        return;
    }
    
    fstat(fd_col_len, &sbcol_len);
    fstat(fd_blk_off, &sbblk_off);

    void *map_base_col_len;
    void *map_base_blk_off;

    map_base_col_len =
          mmap(NULL, sbcol_len.st_size, PROT_READ, MAP_SHARED, fd_col_len, 0);
     
    map_base_blk_off =
          mmap(NULL, sbblk_off.st_size, PROT_READ, MAP_SHARED, fd_blk_off, 0);

    uint64_t num_blk = ((uint64_t)sbblk_off.st_size / sizeof(uint64_t)) - 1;


    //uint64_t chunk_size =  1024 * 8;
    uint64_t out_bytes = chunk_size * num_blk;


    uint8_t* d_out;
    uint64_t real_out_size = ((uint64_t*)(in))[0];

    *out_n_bytes = real_out_size;
    *out = new uint8_t[real_out_size];

    cuda_err_chk(cudaMalloc(&d_out, out_bytes));
    cuda_err_chk(cudaMalloc(&d_in, in_n_bytes));
    cuda_err_chk(cudaMalloc(&d_col_len, sbcol_len.st_size));
    cuda_err_chk(cudaMalloc(&d_blk_offset, sbblk_off.st_size));


    cuda_err_chk(cudaMemcpy(d_in, in, in_n_bytes, cudaMemcpyHostToDevice));

    cuda_err_chk(cudaMemcpy(d_col_len, map_base_col_len, sbcol_len.st_size,
                          cudaMemcpyHostToDevice));

    cuda_err_chk(cudaMemcpy(d_blk_offset, map_base_blk_off, sbblk_off.st_size,
                          cudaMemcpyHostToDevice));

    uint64_t meta_data_bytes = sbcol_len.st_size + sbblk_off.st_size;

    if(munmap(map_base_col_len, sbcol_len.st_size) == -1) PRINT_ERROR;
    if(munmap(map_base_blk_off, sbblk_off.st_size) == -1) PRINT_ERROR;


    dim3 blockD(32,2,1);
    dim3 gridD(num_blk,1,1);
    cudaDeviceSynchronize();

    std::chrono::high_resolution_clock::time_point kernel_start = std::chrono::high_resolution_clock::now();


   // inflate<uint64_t, READ_COL_TYPE, 8 , 8, 4, 512, chunk_size> <<<gridD,blockD>>> (d_in, d_col_len, d_blk_offset, d_out, d_tree, d_slot_struct);
    inflate<uint32_t, DATA_TYPE, uint32_t, QUEUE_SIZE> <<<gridD,blockD>>> (d_in+8, d_out, d_col_len, d_blk_offset, chunk_size, COMP_COL_LEN);
   
    cuda_err_chk(cudaDeviceSynchronize());
    std::chrono::high_resolution_clock::time_point kernel_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total = std::chrono::duration_cast<std::chrono::duration<double>>(kernel_end - kernel_start);
    //std::cout << "kernel time: " << total.count() << " secs\n";
    std::cout << in_n_bytes << "\t" << meta_data_bytes << "\t" <<total.count() << "\n";


    cuda_err_chk(cudaMemcpy((*out), d_out, real_out_size, cudaMemcpyDeviceToHost));

    cuda_err_chk(cudaFree(d_out));
    cuda_err_chk(cudaFree(d_in));
    cuda_err_chk(cudaFree(d_col_len));
    cuda_err_chk(cudaFree(d_blk_offset));
 }

template <typename READ_COL_TYPE, typename DATA_TYPE, typename OUT_COL_TYPE, int QUEUE_SIZE = 8>
__host__ void decompress_1_warp_gpu(const uint8_t* const in, uint8_t** out, const uint64_t in_n_bytes, uint64_t* out_n_bytes, int COMP_COL_LEN, uint64_t chunk_size){
  // uint64_t* col_len_f, const uint64_t col_n_bytes, uint64_t* blk_offset_f, const uint64_t blk_n_bytes) {



    cudaSetDevice(1);
    uint8_t* d_in;
    uint64_t* d_col_len;
    uint64_t* d_blk_offset;

    std::string file_col_len = "./input_data/rle_v1_col_len.bin";
    std::string file_blk_off = "./input_data/rle_v1_blk_offset.bin";

    const char *filename_col_len = file_col_len.c_str();
    const char *filename_blk_off = file_blk_off.c_str();

    int fd_col_len;
    int fd_blk_off;

    struct stat sbcol_len;
    struct stat sbblk_off;

    if ((fd_col_len = open(filename_col_len, O_RDONLY)) == -1) {
        printf("Fatal Error: Col Len read error\n");
        return;
    }
    if ((fd_blk_off = open(filename_blk_off, O_RDONLY)) == -1) {
        printf("Fatal Error: Block off read error\n");
        return;
    }
    
    fstat(fd_col_len, &sbcol_len);
    fstat(fd_blk_off, &sbblk_off);

    void *map_base_col_len;
    void *map_base_blk_off;

    map_base_col_len =
          mmap(NULL, sbcol_len.st_size, PROT_READ, MAP_SHARED, fd_col_len, 0);
     
    map_base_blk_off =
          mmap(NULL, sbblk_off.st_size, PROT_READ, MAP_SHARED, fd_blk_off, 0);

    uint64_t num_blk = ((uint64_t)sbblk_off.st_size / sizeof(uint64_t)) - 1;


    //uint64_t chunk_size =  1024 * 8;
    uint64_t out_bytes = chunk_size * num_blk;


    uint8_t* d_out;
    uint64_t real_out_size = ((uint64_t*)(in))[0];

    *out_n_bytes = real_out_size;
    *out = new uint8_t[real_out_size];

    cuda_err_chk(cudaMalloc(&d_out, out_bytes));
    cuda_err_chk(cudaMalloc(&d_in, in_n_bytes));
    cuda_err_chk(cudaMalloc(&d_col_len, sbcol_len.st_size));
    cuda_err_chk(cudaMalloc(&d_blk_offset, sbblk_off.st_size));


    cuda_err_chk(cudaMemcpy(d_in, in, in_n_bytes, cudaMemcpyHostToDevice));

    cuda_err_chk(cudaMemcpy(d_col_len, map_base_col_len, sbcol_len.st_size,
                          cudaMemcpyHostToDevice));

    cuda_err_chk(cudaMemcpy(d_blk_offset, map_base_blk_off, sbblk_off.st_size,
                          cudaMemcpyHostToDevice));

    if(munmap(map_base_col_len, sbcol_len.st_size) == -1) PRINT_ERROR;
    if(munmap(map_base_blk_off, sbblk_off.st_size) == -1) PRINT_ERROR;


    dim3 blockD(32,2,1);
    dim3 gridD(num_blk,1,1);
    cudaDeviceSynchronize();

    std::chrono::high_resolution_clock::time_point kernel_start = std::chrono::high_resolution_clock::now();


   // inflate<uint64_t, READ_COL_TYPE, 8 , 8, 4, 512, chunk_size> <<<gridD,blockD>>> (d_in, d_col_len, d_blk_offset, d_out, d_tree, d_slot_struct);
    //inflate<uint32_t, DATA_TYPE, uint32_t, QUEUE_SIZE> <<<gridD,blockD>>> (d_in+8, d_out, d_col_len, d_blk_offset, chunk_size, COMP_COL_LEN);
    dim3 blockD_1warp(32,1,1);
    //inflate_1warp<uint32_t, DATA_TYPE, uint32_t, QUEUE_SIZE> <<<gridD,blockD_1warp>>> (d_in+8, d_out, d_col_len, d_blk_offset, chunk_size, COMP_COL_LEN);

    cuda_err_chk(cudaDeviceSynchronize());
    std::chrono::high_resolution_clock::time_point kernel_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total = std::chrono::duration_cast<std::chrono::duration<double>>(kernel_end - kernel_start);
    //std::cout << "kernel time: " << total.count() << " secs\n";
    std::cout << in_n_bytes << "\t" <<total.count() << "\n";


    cuda_err_chk(cudaMemcpy((*out), d_out, real_out_size, cudaMemcpyDeviceToHost));

    cuda_err_chk(cudaFree(d_out));
    cuda_err_chk(cudaFree(d_in));
    cuda_err_chk(cudaFree(d_col_len));
    cuda_err_chk(cudaFree(d_blk_offset));
 }
}

//#endif // __ZLIB_H__
