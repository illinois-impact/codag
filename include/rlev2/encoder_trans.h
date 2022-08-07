#ifndef _RLEV2_ENCODER_TRANPOSE_H_
#define _RLEV2_ENCODER_TRANPOSE_H_

#include "utils.h"
#include "encoder.h"

namespace rlev2 {
    template<bool should_write, int READ_UNIT, typename COMP_TYPE>
    __global__ void block_encode(INPUT_T* in2, const uint64_t in_n_bytes, 
            uint8_t* out, col_len_t* acc_col_len, blk_off_t* blk_off, uint64_t CHUNK_SIZE) {

	    COMP_TYPE* in = (COMP_TYPE*) in2;
        __shared__ unsigned long long int blk_len;

        uint32_t tid = threadIdx.x;
        uint32_t cid = blockIdx.x;


        if (!should_write) {
            if (tid == 0) {
                blk_len = 0;
                if (cid == 0) {
                    blk_off[0] = 0;
                }
            }
            __syncthreads();
        }
        uint64_t in_start_limit = min((cid + 1) * CHUNK_SIZE, in_n_bytes) / sizeof(COMP_TYPE);
        uint64_t in_start = cid * CHUNK_SIZE / sizeof(COMP_TYPE) + tid * READ_UNIT;

        encode_info<> info;
        info.tid = tid; info.cid = cid;
        

        uint32_t write_off =  (cid + tid == 0) ? 0 : acc_col_len[cid * BLK_SIZE + tid - 1];
        info.output = out + write_off;



        INPUT_T prev_delta;

        uint32_t& num_literals = info.num_literals;
        uint32_t& fix_runlen = info.fix_runlen;
        uint32_t& var_runlen = info.var_runlen;
        INPUT_T *literals = info.literals;

        int curr_read_offset = 0;
        // printf("thread %d with chunksize %ld\n", tid, mychunk_size);
        while (true) {
            if (in_start + curr_read_offset >= in_start_limit) break;
            uint64_t val = in[in_start + curr_read_offset]; curr_read_offset ++;

	    if (curr_read_offset == READ_UNIT) {
                in_start += BLK_SIZE * READ_UNIT;
                curr_read_offset = 0;
            }
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

            INPUT_T curr_delta = val - literals[num_literals - 1];
          //  if(threadIdx.x == 0 && blockIdx.x == 0) printf("cur delta: %llu past delta: %llu \n", (unsigned long long) curr_delta, (unsigned long long) info.deltas[0]);
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
                    
                    prev_delta = 0; 
                    for (uint32_t ii = 0; ii < MINIMUM_REPEAT; ++ii) {
                        literals[ii] = val;
                        info.deltas[ii] = 0;
                    }

                    num_literals = MINIMUM_REPEAT;
                }

		else if(fix_runlen >= MINIMUM_REPEAT){
			
			info.deltas[0] = 0;
		}
                if (info.fix_runlen == MAX_LITERAL_SIZE) {
                    determineEncoding(info);
                }

                continue;
            }

            if (info.fix_runlen >= MINIMUM_REPEAT) {
                if (info.fix_runlen <= MAX_SHORT_REPEAT_LENGTH) {
            //          if(threadIdx.x == 0 && blockIdx.x == 0) printf("short val: %lx\n", val);
			writeShortRepeatValues(info);
                } else {

                    info.is_fixed_delta = true;

                    writeDeltaValues(info);
                }
            }

            if (info.fix_runlen > 0 && info.fix_runlen < MINIMUM_REPEAT && val != literals[num_literals - 1]) {
                info.var_runlen = info.fix_runlen;
                info.fix_runlen = 0;
            }

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

        if (!should_write) {
            acc_col_len[BLK_SIZE * cid + tid] = info.potision;
            auto col_len_4B = static_cast<unsigned long long int>((info.potision + 3) / 4 * 4);
            atomicAdd(&blk_len, col_len_4B);
            
            __syncthreads();
            if (tid == 0) {
                // Block alignment should be 4(decoder's READ_UNIT) * 32 
                blk_len = (blk_len + 127) / 128 * 128;
                blk_off[cid + 1] = blk_len;

            }
        }



    }

    template<bool should_write,  typename COMP_TYPE>
    __global__ void block_encode_orig(INPUT_T* in2, const uint64_t in_n_bytes, 
            uint8_t* out, col_len_t* acc_col_len, blk_off_t* blk_off, uint64_t CHUNK_SIZE, uint64_t READ_UNIT) {

        COMP_TYPE* in = (COMP_TYPE*) in2;
        __shared__ unsigned long long int blk_len;

        uint64_t tid = threadIdx.x;
        uint64_t cid = blockIdx.x;




        if (!should_write) {
            if (tid == 0) {
                blk_len = 0;
                if (cid == 0) {
                    blk_off[0] = 0;
                }
            }
            __syncthreads();
        }


        uint64_t in_start_limit = min((cid * 32 + tid + 1) * CHUNK_SIZE / 32, in_n_bytes) / sizeof(COMP_TYPE);
        uint64_t in_start = cid * CHUNK_SIZE / sizeof(COMP_TYPE) + tid * READ_UNIT;



        encode_info<> info;
        info.tid = tid; info.cid = cid;
        

        uint64_t write_off =  (cid + tid == 0) ? 0 : acc_col_len[cid * BLK_SIZE + tid - 1];

        info.output = out + write_off;



        INPUT_T prev_delta;

        uint32_t& num_literals = info.num_literals;
        uint32_t& fix_runlen = info.fix_runlen;
        uint32_t& var_runlen = info.var_runlen;
        INPUT_T *literals = info.literals;

        int curr_read_offset = 0;
        // printf("thread %d with chunksize %ld\n", tid, mychunk_size);
        while (true) {
            if (in_start + curr_read_offset >= in_start_limit) break;
            uint64_t val = in[in_start + curr_read_offset]; curr_read_offset ++;

            //if(tid == 0 && blockIdx.x == 1) printf("val: %lx\n", val);
            // if (curr_read_offset == READ_UNIT) {
            //     in_start += BLK_SIZE * READ_UNIT;
            //     curr_read_offset = 0;
            // }
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

            INPUT_T curr_delta = val - literals[num_literals - 1];
          //  if(threadIdx.x == 0 && blockIdx.x == 0) printf("cur delta: %llu past delta: %llu \n", (unsigned long long) curr_delta, (unsigned long long) info.deltas[0]);
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
                    
                    prev_delta = 0; 
                    for (uint32_t ii = 0; ii < MINIMUM_REPEAT; ++ii) {
                        literals[ii] = val;
                        info.deltas[ii] = 0;
                    }

                    num_literals = MINIMUM_REPEAT;
                }

            else if(fix_runlen >= MINIMUM_REPEAT){
                info.deltas[0] = 0;
             }
                if (info.fix_runlen == MAX_LITERAL_SIZE) {
                    determineEncoding(info);
                }

                continue;
            }

            if (info.fix_runlen >= MINIMUM_REPEAT) {
                if (info.fix_runlen <= MAX_SHORT_REPEAT_LENGTH) {
            //          if(threadIdx.x == 0 && blockIdx.x == 0) printf("short val: %lx\n", val);
            writeShortRepeatValues(info);
                } else {

                    info.is_fixed_delta = true;

                    writeDeltaValues(info);
                }
            }

            if (info.fix_runlen > 0 && info.fix_runlen < MINIMUM_REPEAT && val != literals[num_literals - 1]) {
                info.var_runlen = info.fix_runlen;
                info.fix_runlen = 0;
            }

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

        if (!should_write) {
            acc_col_len[BLK_SIZE * cid + tid] = info.potision;

            blk_off[BLK_SIZE * cid + tid + 1] = (info.potision + 127) / 128 * 128;
        }



    }

    __global__
    void tranpose_col_len_single(uint8_t* in, col_len_t *acc_col_len, col_len_t *col_len, blk_off_t *blk_off, uint8_t* out) {
        uint32_t cid = blockIdx.x;



        col_len_t loc_col_len[32];
        memcpy(loc_col_len, col_len + cid * BLK_SIZE, 32 * sizeof(col_len_t));

        // More space should be saved. TODO: 
        uint64_t curr_iter_off = 0;
        uint64_t out_idx = blk_off[cid];

        while (true) {
            int res = 0;
            for (int tid=0; tid<32; ++tid) {
                if (loc_col_len[tid] > curr_iter_off) {
                    auto tidx = ((cid + tid == 0) ? 0 : acc_col_len[cid * BLK_SIZE + tid - 1]) + curr_iter_off;
                    
                    for (int i=0; i<DECODE_UNIT; ++i) {
                        out[out_idx + i] = in[tidx + i];
                    }

                    out_idx += DECODE_UNIT;
                    res ++;
                }
            }
            if (res == 0) break;
            curr_iter_off += DECODE_UNIT;
        }


    }

        __global__
    void orig_col_len_single(uint8_t* in, col_len_t *acc_col_len, col_len_t *col_len, blk_off_t *blk_off, uint8_t* out) {
        
        uint32_t cid = blockIdx.x;
        col_len_t cur_col_len_offset = 0;
        if(cid !=0) cur_col_len_offset = acc_col_len[cid - 1];

        blk_off_t blk_off_offset = blk_off[cid];

        col_len_t cur_col_len = acc_col_len[cid];
        if(cid != 0) cur_col_len -= acc_col_len[cid - 1];

        for(int i = 0; i < cur_col_len; i++){
            out[blk_off_offset++] = in[cur_col_len_offset++];
        }

    }

    template <int READ_UNIT, typename COMP_TYPE>
    __host__
    void compress_gpu_transpose(const INPUT_T* const in, const uint64_t in_n_bytes, uint8_t*& out, uint64_t& out_n_bytes, uint64_t& meta_data_size,
                    uint64_t& out_n_chunks,blk_off_t *&blk_off, col_len_t *&col_len, uint64_t CHUNK_SIZE) {

	   uint64_t OUTPUT_CHUNK_SIZE = 1.5 * CHUNK_SIZE;
        const uint64_t padded_n_bytes = ((in_n_bytes - 1) / CHUNK_SIZE + 1) * CHUNK_SIZE;

        uint32_t n_chunks = (padded_n_bytes - 1) / CHUNK_SIZE + 1;
        out_n_chunks = n_chunks;
        
        INPUT_T *d_in;
        uint8_t *d_out, *d_out_transpose;
        col_len_t *d_col_len, *d_acc_col_len; //accumulated col len 
        blk_off_t *d_blk_off;
        meta_data_size =  sizeof(col_len_t) * n_chunks * BLK_SIZE +   sizeof(blk_off_t) * (n_chunks + 1);
	    cuda_err_chk(cudaMalloc(&d_in, padded_n_bytes));
	    cuda_err_chk(cudaMemset(d_in, 0, padded_n_bytes));
        cuda_err_chk(cudaMalloc(&d_out, n_chunks * OUTPUT_CHUNK_SIZE));
	    cuda_err_chk(cudaMalloc(&d_col_len, sizeof(col_len_t) * n_chunks * BLK_SIZE));
	    cuda_err_chk(cudaMalloc(&d_acc_col_len, sizeof(col_len_t) * n_chunks * BLK_SIZE));
	    cuda_err_chk(cudaMalloc(&d_blk_off, sizeof(blk_off_t) * (n_chunks + 1)));

	    cuda_err_chk(cudaMemcpy(d_in, in, in_n_bytes, cudaMemcpyHostToDevice));
        
        initialize_bit_maps();

        block_encode<false, READ_UNIT, COMP_TYPE><<<n_chunks, BLK_SIZE>>>(d_in, padded_n_bytes, 
                d_out, d_col_len,  d_blk_off, CHUNK_SIZE);

	    cuda_err_chk(cudaDeviceSynchronize()); 

        thrust::inclusive_scan(thrust::device, d_blk_off, d_blk_off + n_chunks + 1, d_blk_off);
	    cuda_err_chk(cudaDeviceSynchronize()); 
        
        thrust::inclusive_scan(thrust::device, d_col_len, d_col_len + n_chunks * BLK_SIZE, d_acc_col_len);
	    cuda_err_chk(cudaDeviceSynchronize()); 

        block_encode<true, READ_UNIT, COMP_TYPE><<<n_chunks, BLK_SIZE>>>(d_in, padded_n_bytes, 
                            d_out, d_acc_col_len,  d_blk_off, CHUNK_SIZE);
	    cuda_err_chk(cudaDeviceSynchronize()); 

        
        col_len = new col_len_t[n_chunks * BLK_SIZE];
	    cuda_err_chk(cudaMemcpy(col_len, d_col_len, sizeof(col_len_t) * BLK_SIZE * n_chunks, cudaMemcpyDeviceToHost));

        blk_off = new blk_off_t[n_chunks + 1];
	    cuda_err_chk(cudaMemcpy(blk_off, d_blk_off, sizeof(blk_off_t) * (n_chunks + 1), cudaMemcpyDeviceToHost));

        out_n_bytes = blk_off[n_chunks];
        out = new uint8_t[out_n_bytes];
        blk_off[n_chunks] = in_n_bytes; //use last index of blk_off to store file size.
        
        // printf("out n bytes encoding: %lu\n", out_n_bytes);
        cuda_err_chk(cudaFree(d_in));

        cuda_err_chk(cudaMalloc(&d_out_transpose, out_n_bytes));
        tranpose_col_len_single<<<n_chunks, 1>>>(d_out, d_acc_col_len, d_col_len, d_blk_off, d_out_transpose);
        // post_data_tranpose<<<n_chunks, BLK_SIZE>>>(d_out, d_col_len, d_blk_off, d_out_transpose);
        cuda_err_chk(cudaDeviceSynchronize()); 


	    cuda_err_chk(cudaMemcpy(out, d_out_transpose, out_n_bytes, cudaMemcpyDeviceToHost));
        
	    cuda_err_chk(cudaFree(d_out));
	    cuda_err_chk(cudaFree(d_out_transpose));
	    cuda_err_chk(cudaFree(d_col_len));
	    cuda_err_chk(cudaFree(d_acc_col_len));
	    cuda_err_chk(cudaFree(d_blk_off));
    }

    template <int READ_UNIT, typename COMP_TYPE>
    __host__
    void compress_gpu_orig(const INPUT_T* const in, const uint64_t in_n_bytes, uint8_t*& out, uint64_t& out_n_bytes, uint64_t& meta_data_size,
                    uint64_t& out_n_chunks,blk_off_t *&blk_off, col_len_t *&col_len, uint64_t CHUNK_SIZE) {

       uint64_t OUTPUT_CHUNK_SIZE = 1.5* CHUNK_SIZE;
       // const uint64_t padded_n_bytes = ((in_n_bytes - 1) / CHUNK_SIZE + 1) * CHUNK_SIZE;

        uint32_t n_chunks = ((in_n_bytes + CHUNK_SIZE*32 - 1) / (CHUNK_SIZE*32)) * 32;
        const uint64_t padded_n_bytes = n_chunks * CHUNK_SIZE;
        //uint32_t n_chunks = (padded_n_bytes - 1) / CHUNK_SIZE + 1;



        out_n_chunks = n_chunks;
        
        INPUT_T *d_in;
        uint8_t *d_out, *d_out_transpose;
        col_len_t *d_col_len, *d_acc_col_len; //accumulated col len 
        blk_off_t *d_blk_off;
        meta_data_size =  sizeof(col_len_t) * n_chunks * BLK_SIZE +   sizeof(blk_off_t) * (n_chunks + 1);
        cuda_err_chk(cudaMalloc(&d_in, padded_n_bytes));
        cuda_err_chk(cudaMemset(d_in, 0, padded_n_bytes));
        cuda_err_chk(cudaMalloc(&d_out, n_chunks * OUTPUT_CHUNK_SIZE));
        cuda_err_chk(cudaMalloc(&d_col_len, sizeof(col_len_t) * n_chunks * BLK_SIZE));
        cuda_err_chk(cudaMalloc(&d_acc_col_len, sizeof(col_len_t) * n_chunks * BLK_SIZE));
        cuda_err_chk(cudaMalloc(&d_blk_off, sizeof(blk_off_t) * (n_chunks + 1)));

        cuda_err_chk(cudaMemcpy(d_in, in, in_n_bytes, cudaMemcpyHostToDevice));
        
        initialize_bit_maps();

        // block_encode_orig<false, COMP_TYPE><<<n_chunks/32 , BLK_SIZE>>>(d_in, padded_n_bytes, 
        //         d_out, d_col_len,  d_blk_off, CHUNK_SIZE * 32, CHUNK_SIZE);

              block_encode_orig<false, COMP_TYPE><<<n_chunks/32 , BLK_SIZE>>>(d_in, padded_n_bytes, 
                d_out, d_col_len,  d_blk_off, CHUNK_SIZE * 32, CHUNK_SIZE/ sizeof(COMP_TYPE));

        cuda_err_chk(cudaDeviceSynchronize()); 

        thrust::inclusive_scan(thrust::device, d_blk_off, d_blk_off + n_chunks + 1, d_blk_off);
        cuda_err_chk(cudaDeviceSynchronize()); 
        
        thrust::inclusive_scan(thrust::device, d_col_len, d_col_len + n_chunks * BLK_SIZE, d_acc_col_len);
        cuda_err_chk(cudaDeviceSynchronize()); 

        //block_encode_orig<true, COMP_TYPE><<<n_chunks /32 , BLK_SIZE>>>(d_in, padded_n_bytes, 
        //                    d_out, d_acc_col_len,  d_blk_off, CHUNK_SIZE * 32, CHUNK_SIZE);

               block_encode_orig<true, COMP_TYPE><<<n_chunks/32 , BLK_SIZE>>>(d_in, padded_n_bytes, 
                            d_out, d_acc_col_len,  d_blk_off, CHUNK_SIZE * 32, CHUNK_SIZE / sizeof(COMP_TYPE));
        cuda_err_chk(cudaDeviceSynchronize()); 

        
        col_len = new col_len_t[n_chunks * BLK_SIZE];
        cuda_err_chk(cudaMemcpy(col_len, d_col_len, sizeof(col_len_t) * BLK_SIZE * n_chunks, cudaMemcpyDeviceToHost));


        blk_off = new blk_off_t[n_chunks + 1];
        cuda_err_chk(cudaMemcpy(blk_off, d_blk_off, sizeof(blk_off_t) * (n_chunks + 1), cudaMemcpyDeviceToHost));

        out_n_bytes = blk_off[n_chunks];
        out = new uint8_t[out_n_bytes];
        blk_off[n_chunks] = in_n_bytes; //use last index of blk_off to store file size.
        
         cuda_err_chk(cudaFree(d_in));

        cuda_err_chk(cudaMalloc(&d_out_transpose, out_n_bytes));
        orig_col_len_single<<<n_chunks, 1>>>(d_out, d_acc_col_len, d_col_len, d_blk_off, d_out_transpose);
        
        // post_data_tranpose<<<n_chunks, BLK_SIZE>>>(d_out, d_col_len, d_blk_off, d_out_transpose);
        cuda_err_chk(cudaDeviceSynchronize()); 


        cuda_err_chk(cudaMemcpy(out, d_out_transpose, out_n_bytes, cudaMemcpyDeviceToHost));
        
       
        cuda_err_chk(cudaFree(d_out));
        cuda_err_chk(cudaFree(d_out_transpose));
        cuda_err_chk(cudaFree(d_col_len));
        cuda_err_chk(cudaFree(d_acc_col_len));
        cuda_err_chk(cudaFree(d_blk_off));
    }
}

#endif
