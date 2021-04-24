#ifndef __BASE_DEFLATE_H__
#define __BASE_DEFLATE_H__

#include <iostream>
#include <string>
#include <cudf_deflate.cuh>
#include <common.h>
#include <cerrno>
#include <cstring>

#include <iostream>
#include <chrono>

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>

void uncompress(const std::string& in_file, const std::string& out_file)
{
    int in_fd = open(in_file.c_str(), O_RDONLY);
    if (in_fd < 0) {
        std::cerr << "Couldn't open input file: " << in_file << std::endl;
        exit(1);
    }

    struct stat statbuf;
    int err = fstat(in_fd, &statbuf);
    if (err < 0) {
        std::cerr << "Couldn't fstat input file: " << in_file << std::endl;
        exit(2);
    }

    uint64_t * in_ptr = (uint64_t *) mmap(NULL, statbuf.st_size, PROT_READ, MAP_SHARED, in_fd, 0);
    if (in_ptr == MAP_FAILED) {
        std::cerr << "Couldn't mmap input file: " << in_file << std::endl;
        exit(3);
    }
    close(in_fd);

    size_t chunk_size = in_ptr[0];
    size_t n_chunks = in_ptr[1];
    size_t data_offset = 2 + n_chunks;

    printf("chunk size: %llu num chunks: %llu\n", chunk_size, n_chunks);

    uint64_t * sz_arr = in_ptr + 2;


    size_t data_size = statbuf.st_size - (data_offset * sizeof(uint64_t));

    cudf::io::gpu_inflate_input_s * inf_args = new cudf::io::gpu_inflate_input_s[n_chunks];
    //cudf::io::gpu_inflate_status_s * inf_stat = new cudf::io::gpu_inflate_status_s[n_chunks];

    char * d_in;
    char * d_out;

    cuda_err_chk(cudaMalloc(&d_in, data_size));
    cuda_err_chk(cudaMalloc(&d_out, n_chunks * chunk_size));

    cuda_err_chk(cudaMemcpy(d_in, &in_ptr[data_offset], data_size, cudaMemcpyHostToDevice));

    //err = munmap(in_ptr, statbuf.st_size);
    size_t cur_off = 0;
    for (size_t i = 0; i < n_chunks; i++) {
        inf_args[i].srcDevice = d_in + cur_off;
        cur_off += sz_arr[i];
        inf_args[i].srcSize = sz_arr[i];
        inf_args[i].dstDevice = d_out + (chunk_size * i);
        inf_args[i].dstSize = chunk_size;
    }
    err = munmap(in_ptr,statbuf.st_size);
    cudf::io::gpu_inflate_input_s * d_inf_args;
    cudf::io::gpu_inflate_status_s * d_inf_stat;

    cuda_err_chk(cudaMalloc(&d_inf_args, sizeof(cudf::io::gpu_inflate_input_s) * n_chunks));
    cuda_err_chk(cudaMemcpy(d_inf_args, inf_args, sizeof(cudf::io::gpu_inflate_input_s) * n_chunks, cudaMemcpyHostToDevice));
    cuda_err_chk(cudaMalloc(&d_inf_stat, sizeof(cudf::io::gpu_inflate_status_s) * n_chunks));




    std::chrono::high_resolution_clock::time_point kernel_start = std::chrono::high_resolution_clock::now();

    cuda_err_chk(gpuinflate(d_inf_args, d_inf_stat, n_chunks, 1));
    cuda_err_chk(cudaDeviceSynchronize());
	
    std::chrono::high_resolution_clock::time_point kernel_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total = std::chrono::duration_cast<std::chrono::duration<double>>(kernel_end - kernel_start);
    std::cout << "kernel time: " << total.count() << " secs\n";

    size_t out_size = 0;
    cudf::io::gpu_inflate_status_s * inf_stat = new cudf::io::gpu_inflate_status_s[n_chunks];
    cuda_err_chk(cudaMemcpy(inf_stat, d_inf_stat, sizeof(cudf::io::gpu_inflate_status_s) * n_chunks, cudaMemcpyDeviceToHost));

    out_size = (n_chunks - 1) * chunk_size;
    out_size += inf_stat[n_chunks - 1].bytes_written;

    int out_fd = open(out_file.c_str(), O_RDWR | O_CREAT);
    if (out_fd < 0) {
        std::cerr << "Couldn't open output file: " << out_file << std::endl;
        exit(1);
    }

    err = ftruncate(out_fd, out_size);
    if (err < 0) {
        std::cerr << "Couldn't ftruncate output file: " << out_file << std::endl;
        exit(1);
    }

    uint64_t * out_ptr = (uint64_t *) mmap(NULL, out_size, PROT_READ | PROT_WRITE, MAP_SHARED, out_fd, 0);
    if (out_ptr == MAP_FAILED) {
        std::cerr << "Couldn't mmap output file: " << out_file << std::endl;
        std::cerr << std::strerror(errno) << std::endl;
        exit(3);
    }

    close(out_fd);

    cuda_err_chk(cudaMemcpy(out_ptr, d_out, out_size, cudaMemcpyDeviceToHost));

    err = munmap(out_ptr, out_size);


}

#endif // __BASE_DEFLATE_H__
