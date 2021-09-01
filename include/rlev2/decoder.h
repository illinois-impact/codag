#ifndef _RLEV2_DECODER_H_
#define _RLEV2_DECODER_H_
#include "cuda_profiler_api.h"
#include <common.h>
#include <iostream>
#include "utils.h"

#include <chrono>

namespace rlev2 {

    __host__ __device__ inline void write_val(const INPUT_T val, INPUT_T* &out) {
        *(out ++) = val;
    }

    __host__ __device__ inline uint8_t read_byte(uint8_t* &in) {
        return *(in ++);
    }

    __host__ __device__ INPUT_T read_long(uint8_t* &in, uint8_t num_bytes) {
        INPUT_T ret = 0;
        while (num_bytes-- > 0) {
            ret |= ((INPUT_T)read_byte(in) << (num_bytes * 8));
        }
        return ret;
    }

    __host__ __device__ uint32_t decode_short_repeat(uint8_t* &in, uint8_t first, INPUT_T* &out) {
        const uint8_t num_bytes = ((first >> 3) & 0x07) + 1;
        const uint8_t count = (first & 0x07) + MINIMUM_REPEAT;

        const auto val = read_long(in, num_bytes);

        for (uint8_t i=0; i<count; ++i) {
            write_val(val, out);
        }
        
        return num_bytes;
    }

    __host__ __device__ uint32_t readLongs(uint8_t*& in, INPUT_T*& data, uint64_t len, uint8_t fb) {
        uint32_t ret = 0;

        uint64_t bitsLeft = 0;
        uint32_t curByte = 0;
        // TODO: unroll to improve performance
        for(uint64_t i = 0; i < len; i++) {
            uint64_t result = 0;
            uint64_t bitsLeftToRead = fb;
            while (bitsLeftToRead > bitsLeft) {
                result <<= bitsLeft;
                result |= curByte & ((1 << bitsLeft) - 1);
                bitsLeftToRead -= bitsLeft;
                curByte = read_byte(in);
                // printf("read byte %x\n", curByte);
                ++ ret;
                bitsLeft = 8;
            }

            // handle the left over bits
            if (bitsLeftToRead > 0) {
                result <<= bitsLeftToRead;
                bitsLeft -= static_cast<uint32_t>(bitsLeftToRead);
                result |= (curByte >> bitsLeft) & ((1 << bitsLeftToRead) - 1);
            }

            // printf("pb read int %ld(%lu)\n", result, fb);
            write_val(static_cast<int64_t>(result), data);
        }

        return ret;
    }

    __host__ __device__ UINPUT_T readVulong(uint8_t*& in) {
        UINPUT_T ret = 0;
        uint8_t b;
        uint64_t offset = 0;
        do {
            b = read_byte(in);
            ret |= (0x7f & b) << offset;
            offset += 7;
        } while (b >= 0x80);
        return ret;
    }

    __host__ __device__ inline INPUT_T unZigZag(UINPUT_T value) {
        return value >> 1 ^ -(value & 1);
    }
    __host__ __device__ inline INPUT_T readVslong(uint8_t*& in) {
        return unZigZag(readVulong(in));
    }


    __host__ __device__ uint32_t decode_delta(uint8_t* &in, uint8_t first, INPUT_T* &out) {
        uint8_t *in_begin = in;
        const uint8_t encoded_fbw = (first >> 1) & 0x1f;
        const uint8_t fbw = get_decoded_bit_width(encoded_fbw);
        const uint16_t len = ((static_cast<uint16_t>(first & 0x01) << 8) | read_byte(in)) + 1;

        auto base_val = static_cast<INPUT_T>(readVulong(in));
        auto base_delta = static_cast<INPUT_T>(readVslong(in));

        // printf("base val: %ld\n", base_val);
        // printf("base_deltal: %ld\n", base_delta);
        // printf("base len: %u\n", len);

        write_val(base_val, out);
        base_val += base_delta;
        write_val(base_val, out);
        // printf("encoded_fbw(delta) %u(%ld)\n", encoded_fbw, base_delta);

        auto curr = out;

        bool increasing = (base_delta > 0);
        if (encoded_fbw != 0) {
            uint32_t consumed = readLongs(in, out, len - 2, fbw);
            if (increasing) {
                for (uint16_t i=0; i<len; ++i) {
                    base_val = curr[i] += base_val;
                }
            } else {
                for (uint16_t i=0; i<len; ++i) {
                    base_val = curr[i] = base_val - curr[i];
                }
            }
        } else {
            // printf("write fixed delta\n");
            for (uint16_t i=2; i<len; ++i) {
                base_val += base_delta;
                write_val(base_val, out);
            }
        }
        return in - in_begin;
    }

    __host__ __device__ uint32_t decode_direct(uint8_t* &in, uint8_t first, INPUT_T* &out) {
        // printf("Decode DIRECT\n");
        const uint8_t encoded_fbw = (first >> 1) & 0x1f;
        const uint8_t fbw = get_decoded_bit_width(encoded_fbw);
        const uint16_t len = ((static_cast<uint16_t>(first & 0x01) << 8) | read_byte(in)) + 1;

        return 1 + readLongs(in, out, len, fbw);
    }

    __host__ __device__ uint32_t decode_patched_base(uint8_t* &in, uint8_t first, INPUT_T* &out) {
        uint8_t *in_begin = in;

        // printf("Decode PATCHED BASE =>>>>>>>>>>>>>>>>>>>>>.\n\n");
        const uint8_t encoded_fbw = (first >> 1) & 0x1f;
        const uint8_t fbw = get_decoded_bit_width(encoded_fbw);
        const uint16_t len = ((static_cast<uint16_t>(first & 0x01) << 8) | read_byte(in)) + 1;

        const auto third = read_byte(in);
        const auto forth = read_byte(in);

        const uint8_t bw = ((third >> 5) & 0x07) + 1;
        const uint8_t pw = get_decoded_bit_width(third & 0x1f);
        const uint8_t pgw = ((forth >> 5) & 0x07) + 1;
        const uint8_t pll = forth & 0x1f;

        const auto base  = read_long(in, bw);
        // printf("pw base: %ld\n", base);

        uint32_t consumed = 4 + 2; // 4 bytes header + 2 byte long

        auto curr = out;
        consumed += readLongs(in, out, len, fbw);

        uint32_t cfb = get_closest_bit(pw + pgw);

        // TODO Patched List should be no more than 10 percent of MAX_LIT_SIZE
        INPUT_T patch_list[MAX_LITERAL_SIZE], *p_list = patch_list;
        consumed += readLongs(in, p_list, pll, cfb);

        const uint16_t patch_mask = (static_cast<uint16_t>(1) << pw) - 1; // get rid of patch gap
        int gap_idx = 0;
        for (uint8_t p=0; p<pll; ++p) {
            gap_idx += patch_list[p] >> pw;
            curr[gap_idx] |= static_cast<INPUT_T>(patch_list[p] & patch_mask) << fbw;

            // printf("curr[gap_idx]: %ld\n", curr[gap_idx]);
        }

        for (uint16_t i=0; i<len; ++i) {
            // printf("curr[i]: %ld\n",curr[i]);
            curr[i] += base;
        }
        return in - in_begin;
    }

    __host__ __device__ void block_decode(const uint64_t tid, uint8_t* in, const uint64_t* offsets, INPUT_T* out) {
        uint32_t consumed = 0;

        auto curr_out = (INPUT_T*)((uint8_t*)out + tid * CHUNK_SIZE);

        // printf("tid with write offset %ld\n", curr_out - out);

        uint8_t* curr_in = in + offsets[tid];
        const uint32_t my_chunk_size = static_cast<uint32_t>(offsets[tid + 1] - offsets[tid]);
        // printf("tid %llu with chunk size offset %u\n", my_chunk_size);

        // printf("%lu: in(%lu)\n", offsets[tid]);
        while (consumed < my_chunk_size) {
            const auto first = read_byte(curr_in); consumed ++;

            const auto encoding = static_cast<uint8_t>(first & 0xC0); // 0bxx000000
            switch(encoding) {
            case HEADER_SHORT_REPEAT:
                consumed += decode_short_repeat(curr_in, first, curr_out);
                break;
            case HEADER_DIRECT:
                consumed += decode_direct(curr_in, first, curr_out);
                break;
            case HEADER_PACTED_BASE:
                consumed += decode_patched_base(curr_in, first, curr_out);
                break;
            case HEADER_DELTA:  
                consumed += decode_delta(curr_in, first, curr_out);
                break;
            }
            // printf("%llu consumed: %u(%u)\n", consumed, my_chunk_size);
        }
    }

    __global__ void kernel_decompress(uint8_t* in, const uint64_t *ptr, const uint32_t n_chunks, INPUT_T* out) {
        uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < n_chunks) block_decode(tid, in, ptr, out);
    }

    __host__ void decompress_gpu(const uint8_t* in, const uint64_t in_n_bytes, INPUT_T*& out, uint64_t& out_n_bytes) {
	cudaProfilerStart();    
	    initialize_bit_maps();
        
        uint8_t *d_in;
        INPUT_T *d_out;
        uint64_t *d_ptr;
        
        uint32_t n_ptr = *((uint32_t*)in);
        const uint64_t header_bytes = sizeof(uint32_t) + sizeof(uint64_t) * n_ptr + sizeof(uint64_t);
        const uint64_t data_bytes = in_n_bytes - header_bytes;
        const uint64_t raw_data_bytes = *((uint64_t*)(in + header_bytes - sizeof(uint64_t)));

        // printf("ptrs %u\n", n_ptr);
        cuda_err_chk(cudaMalloc((void**)&d_in, data_bytes));
        cuda_err_chk(cudaMalloc((void**)&d_ptr, sizeof(uint64_t) * n_ptr));
        cuda_err_chk(cudaMalloc((void**)&d_out, raw_data_bytes));

        cuda_err_chk(cudaMemcpy(d_in, in + header_bytes, data_bytes, cudaMemcpyHostToDevice));
        cuda_err_chk(cudaMemcpy(d_ptr, in + sizeof(uint32_t), sizeof(uint64_t) * n_ptr, cudaMemcpyHostToDevice));

        uint64_t grid_size = ceil<uint64_t>(n_ptr - 1, BLK_SIZE);
		std::chrono::high_resolution_clock::time_point kernel_start = std::chrono::high_resolution_clock::now();

        kernel_decompress<<<grid_size, BLK_SIZE>>>(d_in, d_ptr, n_ptr - 1, d_out);

	    cuda_err_chk(cudaDeviceSynchronize());
        std::chrono::high_resolution_clock::time_point kernel_end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> total = std::chrono::duration_cast<std::chrono::duration<double>>(kernel_end - kernel_start);		
		std::cout << "kernel time: " << total.count() << " secs\n";

        out = new INPUT_T[raw_data_bytes / sizeof(INPUT_T)];
	    cuda_err_chk(cudaMemcpy(out, d_out, raw_data_bytes, cudaMemcpyDeviceToHost));

        cuda_err_chk(cudaFree(d_in));
        cuda_err_chk(cudaFree(d_out));
        cuda_err_chk(cudaFree(d_ptr));

        out_n_bytes = raw_data_bytes;

	cudaProfilerStop();
	cudaDeviceReset();
    }

}

#endif
