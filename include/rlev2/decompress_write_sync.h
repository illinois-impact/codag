#ifndef _RLEV2_DECODER_TRANPOSE_SYNC_H_
#define _RLEV2_DECODER_TRANPOSE_SYNC_H_

#include "utils.h"

// constexpr int DECODE_BUFFER_COUNT = 256;
// constexpr int SHM_BUFFER_COUNT = DECODE_BUFFER_COUNT * BLK_SIZE;
//constexpr int DECODE_BUFFER_COUNT = 256;
//constexpr int SHM_BUFFER_COUNT = DECODE_BUFFER_COUNT * BLK_SIZE;

#include <cuda/atomic>

#define WRITE_WRAPPING_SIZE 128

namespace rlev2 {
    __global__ void decompress_func_write_sync(const uint8_t* __restrict__ in, const uint64_t n_chunks, const blk_off_t* __restrict__ blk_off, const col_len_t* __restrict__ col_len, int64_t* __restrict__ out) {
        __shared__ cuda::atomic<uint8_t,  cuda::thread_scope_block> out_head_[32];
  		__shared__ cuda::atomic<uint8_t,  cuda::thread_scope_block> out_tail_[32];

		__shared__ int64_t out_[WRITE_WRAPPING_SIZE][BLK_SIZE];

		auto tid = threadIdx.x;
		auto cid = blockIdx.x;

        uint64_t out_off = cid * CHUNK_SIZE / sizeof(int64_t) + tid;

        if(threadIdx.y  == 0){
	    	out_head_[tid] = 0;
	    	out_tail_[tid] = 0;
	    }
	    __syncthreads();

        if (threadIdx.y == 0) {
            uint64_t mychunk_size = col_len[cid * BLK_SIZE + tid];
            uint64_t in_start_idx = blk_off[cid];


            uint32_t* in_4B = (uint32_t *)(&(in[in_start_idx]));
            uint32_t in_4B_off = 0;

            __shared__ uint8_t shm_buffer[SHM_BUFFER_COUNT];
            uint8_t *input_buffer = &shm_buffer[DECODE_BUFFER_COUNT * tid];

            uint8_t input_buffer_head = 0;
            uint8_t input_buffer_tail = 0;
            uint8_t input_buffer_count = 0;

            uint8_t curr_fbw = 0, curr_fbw_left = 0;

            
            uint8_t first_byte;
            uint8_t bits_left_over; // leftover from last byte since bit packing
            bool read_first = false;
             
             int64_t curr_64; // current outint, might not be complete
            uint16_t bits_left = 0;

           
            // for complicated encoding schemes, input_buffer may not have enough data
            // needs a structure to hold the info of last encoding scheme
            int curr_len = 0; 
           
            bool dal_read_base = false;
            int64_t base_val, base_delta, *base_out = out; // for delta encoding

            int bw, pw, pgw, pll, patch_gap, curr_pwb_left; // for patched base encoding

            uint64_t used_bytes = 0;

            auto read_byte = [&]() {
// #ifdef DEBUG
// if (cid == ERR_CHUNK && tid == ERR_THREAD) printf("thread %d read byte %x\n", tid, input_buffer[input_buffer_head]);
// #endif
                auto ret = input_buffer[input_buffer_head];
                input_buffer_count -= 1;
                used_bytes += 1;
                input_buffer_head = (input_buffer_head + 1) % DECODE_BUFFER_COUNT;
                return ret;
            };

            auto write_int = [&](int64_t i) {
 
                auto curr_tail = out_tail_[tid].load(cuda::memory_order_relaxed);
                auto next_tail = (curr_tail + 1) % WRITE_WRAPPING_SIZE;
                while (next_tail == out_head_[tid].load(cuda::memory_order_acquire)) {
                    __nanosleep(100);
                }
                out_[curr_tail][tid] = i;

// #ifdef DEBUG
// if (cid == 0 && tid == ERR_THREAD) printf("thread %d write int %lld\n", tid, i);
// #endif
                out_tail_[tid].store(next_tail, cuda::memory_order_release);
                curr_len --;
                curr_64 = 0;
                curr_fbw_left = curr_fbw;
            };

            auto read_uvarint = [&]() {
                uint64_t out_int = 0;
                int offset = 0;
                uint8_t b = 0;
                do {
                    b = read_byte();
                    out_int |= (VARINT_MASK & b) << offset;
                    offset += 7;
                } while (b >= 0x80);
                return out_int;
            };

            auto read_svarint = [&]() {
                auto ret = static_cast<int64_t>(read_uvarint());
                return ret >> 1 ^ -(ret & 1);
            };

            auto read_long = [&](uint8_t fbw) {
                int64_t ret = 0;
                while (fbw-- > 0) {
                    ret |= (read_byte() << (fbw * 8));
                }
                return ret;
            };

            const uint32_t t_read_mask = (0xffffffff >> (32 - tid));
            while (used_bytes < mychunk_size) {
                auto mask = __activemask();
                bool read;
                #pragma unroll
                for (int read_iter=0; read_iter<2; ++read_iter) {
                    
                    read = used_bytes + input_buffer_count < mychunk_size;
                    uint32_t read_sync = __ballot_sync(mask, read);
                    if (read) {
// if (cid == 0 && tid == ERR_THREAD) {
//     printf("thread %d with %d threads before active\n", tid, __popc(read_sync & t_read_mask));
// }
                        *(uint32_t *)(&(input_buffer[input_buffer_tail])) = in_4B[in_4B_off + __popc(read_sync & t_read_mask)];  
// if (cid == 0 && tid == ERR_THREAD) {
//     printf("thread %d read bytes %x%x%x%x\n", tid,
//      input_buffer[input_buffer_tail], 
//       input_buffer[input_buffer_tail + 1], 
//        input_buffer[input_buffer_tail + 2], 
//         input_buffer[input_buffer_tail + 3]);
// }
                        input_buffer_tail = (input_buffer_tail + 4) % DECODE_BUFFER_COUNT;

                        input_buffer_count += 4;
                        in_4B_off += __popc(read_sync); 
                    } 
                    __syncwarp(mask);
                }


                if (!read_first) {
                    read_first = true;
                    first_byte = read_byte();
                    curr_fbw = get_decoded_bit_width((first_byte >> 1) & 0x1f);
                    curr_fbw_left = curr_fbw;
                    
                    if ((first_byte & HEADER_MASK) != HEADER_SHORT_REPEAT) {
                        curr_len = (((first_byte & 0x01) << 8) | read_byte()) + 1;
                        bits_left = 0; bits_left_over = 0; curr_64 = 0;

                        dal_read_base = false;
                        
                        if ((first_byte & HEADER_MASK) == HEADER_PACTED_BASE) {
                            auto third = read_byte();
                            auto forth = read_byte();

                            bw = ((third >> 5) & 0x07) + 1;
                            pw = get_decoded_bit_width(third & 0x1f);
                            pgw = ((forth >> 5) & 0x07) + 1;
                            pll = forth & 0x1f;
                            patch_gap = 0;

                            curr_pwb_left = get_closest_bit(pw + pgw);
                            dal_read_base = false;
                        }
                    }
                }

                switch(first_byte & HEADER_MASK) {
                case HEADER_SHORT_REPEAT: {
                    auto num_bytes = ((first_byte >> 3) & 0x07) + 1;
                    if (num_bytes <= input_buffer_count) { 
                        int64_t tmp_int = 0;
                        while (num_bytes-- > 0) {
                            tmp_int |= ((int64_t)read_byte() << (num_bytes * 8));
                        }
                        auto cnt = (first_byte & 0x07) + MINIMUM_REPEAT;
                        while (cnt-- > 0) {
                            write_int(tmp_int);
                        }
                        read_first = false;
                    } 
                }	break;
                case HEADER_DIRECT: {
                    while (curr_len > 0) {
                        while (curr_fbw_left > bits_left) {
                            if (input_buffer_count == 0) goto main_loop;
                            curr_64 <<= bits_left;
                            curr_64 |= bits_left_over & ((1 << bits_left) - 1);
                            curr_fbw_left -= bits_left;
                            bits_left_over = read_byte();
                            bits_left = 8;
                        }

                        if (curr_fbw_left <= bits_left) {
                            if (curr_fbw_left > 0) {
                                curr_64 <<= curr_fbw_left;
                                bits_left -= curr_fbw_left;
                                curr_64 |= (bits_left_over >> bits_left) & ((1 << curr_fbw_left) - 1);
                            }

                            write_int(curr_64);
                        } 
                    }
                    
                    read_first = false;
                }	break;
                case HEADER_DELTA: {
                    if (!dal_read_base) {
                        if (input_buffer_count < 64 + curr_fbw && read) break;
                        base_val = read_uvarint();
                        base_delta = read_svarint();
                        write_int(base_val);
                        base_val += base_delta;
                        write_int(base_val);
                        dal_read_base = true;
                    } 
                    
                    if (dal_read_base) {
                        if (((first_byte >> 1) & 0x1f) != 0) {
                            while (curr_len > 0) {
                                while (curr_fbw_left > bits_left) {
                                    if (input_buffer_count == 0) goto main_loop;
                                    curr_64 <<= bits_left;
                                    curr_64 |= bits_left_over & ((1 << bits_left) - 1);
                                    curr_fbw_left -= bits_left;
                                    bits_left_over = read_byte();
                                    bits_left = 8;
                                }

                                if (curr_fbw_left <= bits_left) {
                                    if (curr_fbw_left > 0) {
                                        curr_64 <<= curr_fbw_left;
                                        bits_left -= curr_fbw_left;
                                        curr_64 |= (bits_left_over >> bits_left) & ((1 << curr_fbw_left) - 1);
                                    }

                                    base_val += curr_64;
                                    write_int(base_val);
                                } 
                            }
                            read_first = false;
                        } else {
                            // fixed delta encoding
                            while (curr_len > 0) {
                                base_val += base_delta;
                                write_int(base_val);
                                curr_len --;
                            }
                            read_first = false;
                        }
                    }
                }	break;
                case HEADER_PACTED_BASE: {
                    if (!dal_read_base) {
                        if (input_buffer_count < bw) goto main_loop;
                        dal_read_base = true;
                        base_val = read_long(bw);
                    } 

                    while (curr_len > 0) {
                        while (curr_fbw_left > bits_left) {
                            if (input_buffer_count == 0) goto main_loop;
                            curr_64 <<= bits_left;
                            curr_64 |= bits_left_over & ((1 << bits_left) - 1);
                            curr_fbw_left -= bits_left;
                            bits_left_over = read_byte();
                            bits_left = 8;
                        }

                        if (curr_fbw_left <= bits_left) {
                            if (curr_fbw_left > 0) {
                                curr_64 <<= curr_fbw_left;
                                bits_left -= curr_fbw_left;
                                curr_64 |= (bits_left_over >> bits_left) & ((1 << curr_fbw_left) - 1);
                            }

                            curr_64 += base_val;
                            write_int(curr_64);
                        } 
                    }

                    auto patch_mask = (static_cast<uint16_t>(1) << pw) - 1;
                    while (pll > 0) {
                        while (curr_pwb_left > bits_left) {
                            if (input_buffer_count == 0) goto main_loop;
                            curr_64 <<= bits_left;
                            curr_64 |= bits_left_over & ((1 << bits_left) - 1);
                            curr_pwb_left -= bits_left;
                            bits_left_over = read_byte();
                            bits_left = 8;
                        }

                        if (curr_pwb_left <= bits_left) {
                            if (curr_pwb_left > 0) {
                                curr_64 <<= curr_pwb_left;
                                bits_left -= curr_pwb_left;
                                curr_64 |= (bits_left_over >> bits_left) & ((1 << curr_pwb_left) - 1);
                            }

                            patch_gap += curr_64 >> pw;
                            base_out[patch_gap * BLK_SIZE] |= static_cast<int64_t>(curr_64 & patch_mask) << curr_fbw;

                            pll --;
                            curr_64 = 0;
                            curr_pwb_left = get_closest_bit(pw + pgw);
                        } 

                    }
                    read_first = false; 
                }	break;
                }
            main_loop:
            __syncwarp(mask);
            }

        }
		else { 
            for(int i = 0; i < CHUNK_SIZE/sizeof(int64_t)/BLK_SIZE; i++) {
// if (cid == 0 && tid == ERR_THREAD) printf("thread %d with outhead(%u) outtail(%u) wit head val(%lld)\n", tid, out_head_[tid].load(), out_tail_[tid].load(), out_[out_head_[tid].load()][tid]);
				r5:
					const auto cur_head = out_head_[tid].load(cuda::memory_order_relaxed);
					if (cur_head == out_tail_[tid].load(cuda::memory_order_acquire)) {
// if (cid == 0 ) {
//     printf("thread %d stuck at worker \n", tid);
// }
     
						__nanosleep(100);
						goto r5;
					}

					auto temp_out = out_[cur_head][tid];
			

					const auto next_head = (cur_head + 1) % WRITE_WRAPPING_SIZE;
					out_head_[tid].store(next_head, cuda::memory_order_release);

					__syncwarp();
					//writing based on col_idx
					out[out_off] = temp_out;
// #ifdef DEBUG
// if (cid == 0 && tid == ERR_THREAD) printf("thread %d write %lld to out\n", tid, temp_out);
// #endif
					out_off += 32;
			}
		
        }

    }

}
#endif