#ifndef _RLEV2_ENCODER_TRANPOSE_H_
#define _RLEV2_ENCODER_TRANPOSE_H_

#include "utils.h"
#include "encoder.h"

#include <cstdint>

#define READ_UNIT 1 //read 4 ints at once
#define WRITE_UNIT 4 //equivalent to "decoder should read 4 bytes at one op"

namespace rlev2 {
    // __global__ 
    __global__
    void block_encode_new(int64_t* in, const uint64_t in_n_bytes, 
            uint8_t* out, col_len_t* col_len, blk_off_t* blk_off) {

        __shared__ unsigned long long int blk_len;
        uint32_t tid = threadIdx.x;
        uint32_t cid = blockIdx.x;

        if (tid == 0) {
            blk_len = 0;
            if (cid == 0) {
                blk_off[0] = 0;
            }
        }
        __syncthreads();
        int64_t in_start_limit = min((cid + 1) * CHUNK_SIZE, in_n_bytes) / sizeof(int64_t);
        int64_t in_start = cid * CHUNK_SIZE / sizeof(int64_t) + tid * READ_UNIT;

        // printf("thread %d: %ld to %ld\n", tid, in_start, in_start_limit);

        encode_info info;
        info.output = out + 32 * tid;

        int64_t prev_delta;

        auto& num_literals = info.num_literals;
        auto& fix_runlen = info.fix_runlen;
        auto& var_runlen = info.var_runlen;
        int64_t *literals = info.literals;

        // printf("thread %d with chunksize %ld\n", tid, mychunk_size);
        while (in_start < in_start_limit) {
            auto val = in[in_start];
            in_start += READ_UNIT * BLK_SIZE;
            // if (tid == 0) printf("%lu read %ld\n", tid, val);
            if (num_literals == 0) {
                literals[num_literals ++] = val;
                fix_runlen = 1;
                var_runlen = 1;
                continue;
            }

            if (num_literals == 1) {
                literals[num_literals ++] = val; 
                prev_delta = val - literals[0];

                if (val == literals[0]) {
                    fix_runlen = 2; var_runlen = 0;
                } else {
                    fix_runlen = 0; var_runlen = 2;
                }
                continue;
            }

            int64_t curr_delta = val - literals[num_literals - 1];
            if (prev_delta == 0 && curr_delta == 0) {
                // fixed run length
                literals[num_literals ++] = val;
                if (var_runlen > 0) {
                    // fix run len is at the end of literals
                    fix_runlen = 2;
                }
                fix_runlen ++;

                if (fix_runlen >= MINIMUM_REPEAT && var_runlen > 0) {
                    num_literals -= MINIMUM_REPEAT;
                    var_runlen -= (MINIMUM_REPEAT - 1);

                    determineEncoding(info);

                    for (uint32_t ii = 0; ii < MINIMUM_REPEAT; ++ii) {
                        literals[ii] = val;
                    }
                    num_literals = MINIMUM_REPEAT;
                }

                if (info.fix_runlen == MAX_LITERAL_SIZE) {
                    determineEncoding(info);
                }

                continue;
            }

            // case 2: variable delta run

            // if fixed run length is non-zero and if it satisfies the
            // short repeat conditions then write the values as short repeats
            // else use delta encoding
            if (info.fix_runlen >= MINIMUM_REPEAT) {
                if (info.fix_runlen <= MAX_SHORT_REPEAT_LENGTH) {
                    writeShortRepeatValues(info);
                } else {
                    info.is_fixed_delta = true;
                    writeDeltaValues(info);
                }
            }

            // if fixed run length is <MINIMUM_REPEAT and current value is
            // different from previous then treat it as variable run
            if (info.fix_runlen > 0 && info.fix_runlen < MINIMUM_REPEAT && val != literals[num_literals - 1]) {
                info.var_runlen = info.fix_runlen;
                info.fix_runlen = 0;
            }

            // after writing values re-initialize the variables
            if (num_literals == 0) {
                literals[num_literals ++] = val;
                fix_runlen = 1;
                var_runlen = 1;
            } else {
                prev_delta = val - literals[num_literals - 1];
                literals[num_literals++] = val;
                info.var_runlen++;

                if (info.var_runlen == MAX_LITERAL_SIZE) {
                    determineEncoding(info);
                }
            }


        }

        // printf("finish reading\n");
        if (num_literals != 0) {
            if (info.var_runlen != 0) {
                determineEncoding(info);
            } else if (info.fix_runlen != 0) {
                if (info.fix_runlen < MINIMUM_REPEAT) {
                    info.var_runlen = info.fix_runlen;
                    info.fix_runlen = 0;
                    determineEncoding(info);
                } else if (info.fix_runlen >= MINIMUM_REPEAT
                        && info.fix_runlen <= MAX_SHORT_REPEAT_LENGTH) {
                    writeShortRepeatValues(info);
                } else {
                    info.is_fixed_delta = true;
                    writeDeltaValues(info);
                }
            }
        }

        col_len[BLK_SIZE * cid + tid] = info.potision;
        auto col_len_4B = static_cast<unsigned long long int>((info.potision + 3) / 4 * 4);
        atomicAdd(&blk_len, col_len_4B);
        
        __syncthreads();
        if (tid == 0) {
            // Block alignment should be 4(decoder's READ_UNIT) * 32 
            blk_len = (blk_len + 127) / 128 * 128;
            blk_off[cid + 1] = blk_len;
        }
        // *offset = info.potision;
        // printf("%lu: %x\n", tid, info.output[0]);
        // printf("%ld: %u\n", tid, info.potision);
    }

    __global__
    void block_encode_new_write(int64_t* in, const uint64_t in_n_bytes, 
            uint8_t* out, col_len_t* acc_col_len, blk_off_t* blk_off) {

        uint32_t tid = threadIdx.x;
        uint32_t cid = blockIdx.x;

        int64_t in_start_limit = min((cid + 1) * CHUNK_SIZE, in_n_bytes) / sizeof(int64_t);
        int64_t in_start = cid * CHUNK_SIZE / sizeof(int64_t) + tid * ENCODE_UNIT;

        //TODO: Make this more intelligent
        // uint8_t* out_4B = blk_off[cid] - blk_off[0] + WRITE_UNIT * tid;

        uint32_t write_off =  (cid + tid == 0) ? 0 : acc_col_len[cid * BLK_SIZE + tid - 1];
        // printf("thread %d write offset %u\n", tid, write_off);

        uint8_t* out_4B = out + write_off;


        encode_info info;
        info.output = out_4B;

        int64_t prev_delta;

        auto& num_literals = info.num_literals;
        auto& fix_runlen = info.fix_runlen;
        auto& var_runlen = info.var_runlen;
        int64_t *literals = info.literals;

        int read_unit = 0;

        // printf("thread %d with chunksize %ld\n", tid, mychunk_size);
        while (in_start < in_start_limit) {
            auto val = in[in_start];
            read_unit ++;
            if (read_unit == ENCODE_UNIT) {
                read_unit = 0;
                in_start += ENCODE_UNIT * BLK_SIZE;
            }
            // if (tid == 0) printf("thread %u read %ld\n", tid, val);
            if (num_literals == 0) {
                literals[num_literals ++] = val;
                fix_runlen = 1;
                var_runlen = 1;
                continue;
            }

            if (num_literals == 1) {
                literals[num_literals ++] = val; 
                prev_delta = val - literals[0];

                if (val == literals[0]) {
                    fix_runlen = 2; var_runlen = 0;
                } else {
                    fix_runlen = 0; var_runlen = 2;
                }
                continue;
            }

            int64_t curr_delta = val - literals[num_literals - 1];
            if (prev_delta == 0 && curr_delta == 0) {
                // fixed run length
                literals[num_literals ++] = val;
                if (var_runlen > 0) {
                    // fix run len is at the end of literals
                    fix_runlen = 2;
                }
                fix_runlen ++;

                if (fix_runlen >= MINIMUM_REPEAT && var_runlen > 0) {
                    num_literals -= MINIMUM_REPEAT;
                    var_runlen -= (MINIMUM_REPEAT - 1);

                    determineEncoding(info);

                    for (uint32_t ii = 0; ii < MINIMUM_REPEAT; ++ii) {
                        literals[ii] = val;
                    }
                    num_literals = MINIMUM_REPEAT;
                }

                if (info.fix_runlen == MAX_LITERAL_SIZE) {
                    determineEncoding(info);
                }

                continue;
            }

            // case 2: variable delta run

            // if fixed run length is non-zero and if it satisfies the
            // short repeat conditions then write the values as short repeats
            // else use delta encoding
            if (info.fix_runlen >= MINIMUM_REPEAT) {
                if (info.fix_runlen <= MAX_SHORT_REPEAT_LENGTH) {
                    writeShortRepeatValues(info);
                } else {
                    info.is_fixed_delta = true;
                    writeDeltaValues(info);
                }
            }

            // if fixed run length is <MINIMUM_REPEAT and current value is
            // different from previous then treat it as variable run
            if (info.fix_runlen > 0 && info.fix_runlen < MINIMUM_REPEAT && val != literals[num_literals - 1]) {
                info.var_runlen = info.fix_runlen;
                info.fix_runlen = 0;
            }

            // after writing values re-initialize the variables
            if (num_literals == 0) {
                // initializeLiterals(val); // REMOVE COMMENT HERE
                literals[num_literals ++] = val;
                fix_runlen = 1;
                var_runlen = 1;
            } else {
                prev_delta = val - literals[num_literals - 1];
                literals[num_literals++] = val;
                info.var_runlen++;

                if (info.var_runlen == MAX_LITERAL_SIZE) {
                    determineEncoding(info);
                }
            }


        }

        // printf("finish reading\n");
        if (num_literals != 0) {
            if (info.var_runlen != 0) {
                determineEncoding(info);
            } else if (info.fix_runlen != 0) {
                if (info.fix_runlen < MINIMUM_REPEAT) {
                    info.var_runlen = info.fix_runlen;
                    info.fix_runlen = 0;
                    determineEncoding(info);
                } else if (info.fix_runlen >= MINIMUM_REPEAT
                        && info.fix_runlen <= MAX_SHORT_REPEAT_LENGTH) {
                    writeShortRepeatValues(info);
                } else {
                    info.is_fixed_delta = true;
                    writeDeltaValues(info);
                }
            }
        }

        // if (cid == 1 && tid == 0) {
        //     for (int i=0; i<acc_col_len[0]; ++i) {
        //         printf("chunk 1 thread 0 writes %x\n", info.output[i]);
        //     }
        // }

        // printf("thread %d write %u bytes\n", tid, info.potision);
    }

    __global__ 
    void tranpose_col_len(uint8_t* in, col_len_t *acc_col_len, blk_off_t *blk_off, uint8_t* out) {
        uint32_t tid = threadIdx.x;
        uint32_t cid = blockIdx.x;

        uint64_t in_idx = (cid + tid == 0) ? 0 : acc_col_len[cid * BLK_SIZE + tid - 1];
        uint64_t out_idx = blk_off[cid] + tid * DECODE_UNIT;
        int64_t out_bytes = acc_col_len[cid * BLK_SIZE + tid] - ((cid + tid == 0) ? 0 : acc_col_len[cid * BLK_SIZE + tid - 1]);

        while (out_bytes > 0) {
            auto mask = __activemask();

            for (int i=0; i<DECODE_UNIT; ++i) {
                out[out_idx + i] = in[in_idx ++];
            }
            
            auto res = __popc(mask);
            out_idx += DECODE_UNIT * res;
            out_bytes -= DECODE_UNIT;

            __syncwarp(mask);
        }
        
    }


    __host__
    void compress_gpu_transpose(const int64_t* const in, const uint64_t in_n_bytes, uint8_t*& out, uint64_t& out_n_bytes,
                    uint64_t& out_n_chunks, blk_off_t *&blk_off, col_len_t *&col_len) {
        uint32_t n_chunks = (in_n_bytes - 1) / CHUNK_SIZE + 1;
        out_n_chunks = n_chunks;
        
        int64_t *d_in;
        uint8_t *d_out, *d_out_transpose;
        col_len_t *d_col_len, *d_acc_col_len; //accumulated col len 
        blk_off_t *d_blk_off;
        
        // printf("input chunk: %lu\n", CHUNK_SIZE);
        // printf("output chunk: %lu\n", n_chunks * OUTPUT_CHUNK_SIZE);

        printf("in_n_bytes: %lu\n", in_n_bytes);
        printf("n_chunks: %u\n", n_chunks);

	    cuda_err_chk(cudaMalloc(&d_in, in_n_bytes));
        cuda_err_chk(cudaMalloc(&d_out, n_chunks * OUTPUT_CHUNK_SIZE));
	    cuda_err_chk(cudaMalloc(&d_col_len, sizeof(col_len_t) * n_chunks * BLK_SIZE));
	    cuda_err_chk(cudaMalloc(&d_acc_col_len, sizeof(col_len_t) * n_chunks * BLK_SIZE));
	    cuda_err_chk(cudaMalloc(&d_blk_off, sizeof(blk_off_t) * (n_chunks + 1)));

	    cuda_err_chk(cudaMemcpy(d_in, in, in_n_bytes, cudaMemcpyHostToDevice));
        
        initialize_bit_maps();

        block_encode_new<<<n_chunks, BLK_SIZE>>>(d_in, in_n_bytes, 
                            d_out, d_col_len,  d_blk_off);
	    cuda_err_chk(cudaDeviceSynchronize()); 

        thrust::inclusive_scan(thrust::device, d_blk_off, d_blk_off + n_chunks + 1, d_blk_off);
	    cuda_err_chk(cudaDeviceSynchronize()); 
        
        thrust::inclusive_scan(thrust::device, d_col_len, d_col_len + n_chunks * BLK_SIZE, d_acc_col_len);
	    cuda_err_chk(cudaDeviceSynchronize()); 

        block_encode_new_write<<<n_chunks, BLK_SIZE>>>(d_in, in_n_bytes, 
                            d_out, d_acc_col_len,  d_blk_off);
	    cuda_err_chk(cudaDeviceSynchronize()); 

        
        col_len = new col_len_t[n_chunks * BLK_SIZE];
	    cuda_err_chk(cudaMemcpy(col_len, d_col_len, sizeof(col_len_t) * BLK_SIZE * n_chunks, cudaMemcpyDeviceToHost));

        blk_off = new blk_off_t[n_chunks + 1];
	    cuda_err_chk(cudaMemcpy(blk_off, d_blk_off, sizeof(blk_off_t) * (n_chunks + 1), cudaMemcpyDeviceToHost));

        out_n_bytes = blk_off[n_chunks];
        out = new uint8_t[out_n_bytes];
        blk_off[n_chunks] = in_n_bytes; //use last index of blk_off to store file size.
        
        cuda_err_chk(cudaMalloc(&d_out_transpose, out_n_bytes));
        tranpose_col_len<<<n_chunks, BLK_SIZE>>>(d_out, d_acc_col_len, d_blk_off, d_out_transpose);
	    cuda_err_chk(cudaDeviceSynchronize()); 


	    cuda_err_chk(cudaMemcpy(out, d_out_transpose, out_n_bytes, cudaMemcpyDeviceToHost));

	    // uint64_t padded_out_size = blk_off[n_chunks - 1];
        
        // printf("eachh thread output %lu bytes\n", col_len[0]);


        // for (int i=0; i<128; ++i) {
        //     printf("out[%d]: %x\n", i, out[i]);
        // }

        // for (int i=0; i<=n_chunks; ++i) {
        //     printf("blk_off[%d]: %lu\n", i, blk_off[i]);
        // }



        // delete[] col_len;
        // delete[] blk_off;
        // delete[] out;


	    cuda_err_chk(cudaFree(d_in));
	    cuda_err_chk(cudaFree(d_out));
	    cuda_err_chk(cudaFree(d_out_transpose));
	    cuda_err_chk(cudaFree(d_col_len));
	    cuda_err_chk(cudaFree(d_acc_col_len));
	    cuda_err_chk(cudaFree(d_blk_off));
    }


    // void compress_func_new(const int64_t* const in, uint8_t* out, 
    //                     const uint64_t in_n_bytes, const uint64_t out_n_bytes, 
    //                     const uint64_t out_chunk_size, const uint64_t n_chunks, 
    //                     const blk_off_t* const blk_off, const col_len_t* const col_map_t, const uint8_t* const col_map) {        // auto tid = threadIdx.x;
    //     int tid = 0;
    // 	int chunk_idx = 0;

    // 	// uint64_t padded_in_n_bytes = in_n_bytes + (CHUNK_SIZE-(in_n_bytes % CHUNK_SIZE));

    //     uint64_t in_start_idx = chunk_idx * CHUNK_SIZE;

    //     uint64_t used_bytes = 0, read_bytes = 0;

    //     col_len_t out_bytes = 0;

    //     while (used_bytes < CHUNK_SIZE) {
            
    //     }

    // }
}

#endif