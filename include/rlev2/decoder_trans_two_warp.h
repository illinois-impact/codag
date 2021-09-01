#ifndef _RLEV2_DECODER_DOUBLE_WARP_H_
#define _RLEV2_DECODER_DOUBLE_WARP_H_

#include <cuda/atomic>
#include "utils.h"

namespace rlev2 {
	template <int READ_UNIT>
	__global__ void decompress_func_read_sync(const uint8_t* __restrict__ in, const uint64_t n_chunks, const blk_off_t* __restrict__ blk_off, const col_len_t* __restrict__ col_len, INPUT_T* __restrict__ out) {
		__shared__ uint8_t in_[WARP_SIZE][DECODE_BUFFER_COUNT];
		__shared__ cuda::atomic<uint8_t, cuda::thread_scope_block> in_cnt_[WARP_SIZE];

		__shared__ INPUT_T out_buffer[WARP_SIZE][WRITE_VEC_SIZE];

		int tid = threadIdx.x;
		int cid = blockIdx.x;
		int which = threadIdx.y;

		uint32_t used_bytes = 0;
		uint32_t mychunk_size = col_len[cid * BLK_SIZE + tid];

		if (which == 0) {
			in_cnt_[tid] = 0;
		}

		__syncthreads();

		if (which == 0) { // reading warp
			uint64_t in_start_idx = blk_off[cid];
			uint32_t* in_4B = (uint32_t *)(&(in[in_start_idx]));
			uint32_t in_4B_off = 0;	

			uint8_t in_tail_ = 0;

			const uint32_t t_read_mask = (0xffffffff >> (32 - tid));
			while (true) {
				bool read = used_bytes < mychunk_size;
				unsigned read_sync = __ballot_sync(0xffffffff, read);

				if (read) {
					while (in_cnt_[tid].load(cuda::memory_order_acquire) + 4 > DECODE_BUFFER_COUNT) {
						__nanosleep(1000);
					}

					__syncwarp(read_sync);

					*(uint32_t*)(&in_[tid][in_tail_]) = in_4B[in_4B_off + __popc(read_sync & t_read_mask)];  
					in_cnt_[tid].fetch_add(4, cuda::memory_order_release);
					in_tail_ = (in_tail_ + 4) % DECODE_BUFFER_COUNT;
					in_4B_off += __popc(read_sync);
					used_bytes += 4;
				} else {
					break;
				}
			}
		} else if (which == 1) { // compute warp
			INPUT_T* out_8B = out + (cid * CHUNK_SIZE / sizeof(INPUT_T) + tid * READ_UNIT);
			INPUT_T* base_out = out + cid * CHUNK_SIZE / sizeof(INPUT_T);
			uint32_t out_ptr = 0;

			uint8_t in_head_ = 0;
			uint8_t *in_ptr_ = &(in_[tid][0]);
			uint8_t out_buffer_ptr = 0;
			uint8_t out_counter = 0;

			auto deque_int = [&]() {
				*reinterpret_cast<VEC_T*>(out_8B + out_counter) = *reinterpret_cast<VEC_T*>(out_buffer[tid]);
				
				out_counter += WRITE_VEC_SIZE;
				if (out_counter == READ_UNIT) {
					out_counter = 0;
					out_8B += BLK_SIZE * READ_UNIT;
				}    
				out_buffer_ptr = 0;
			};

			auto write_int = [&](INPUT_T i) {
				out_ptr ++;
				if (READ_UNIT >= 4) {
					if (out_buffer_ptr == WRITE_VEC_SIZE) {
						deque_int();
					}
					out_buffer[tid][out_buffer_ptr++] = i;
				} else {
#ifdef DEBUG
if (cid == ERR_CHUNK && tid == ERR_THREAD) printf("thread %u write %u at offset %d\n", tid, i, out_8B + out_buffer_ptr - out);
#endif
					*(out_8B + out_buffer_ptr) = i; 
					out_buffer_ptr ++;
					if (out_buffer_ptr == READ_UNIT) {
						out_buffer_ptr = 0;
						out_8B += BLK_SIZE * READ_UNIT;
					}
				}
			};
			
			auto read_byte = [&]() {
				while (in_cnt_[tid].load(cuda::memory_order_acquire) == 0) {
					if (out_buffer_ptr == WRITE_VEC_SIZE) {
						deque_int();
					}
				}

				uint8_t ret = in_ptr_[in_head_];
				in_head_ = (in_head_ + 1) % DECODE_BUFFER_COUNT;
				in_cnt_[tid].fetch_sub(1, cuda::memory_order_release);
				used_bytes += 1;
				return ret;
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
#ifdef DEBUG
if (cid == ERR_CHUNK && tid == ERR_THREAD) {
	printf("thread %u case direct read base delta %d >>>\n", tid, ret);
}
#endif
				return ret >> 1 ^ -(ret & 1);
			};
			
			while (used_bytes < mychunk_size) {
				auto first = read_byte();
				switch(first & HEADER_MASK) {
				case HEADER_SHORT_REPEAT: {
					auto nbytes = ((first >> 3) & 0x07) + 1;
					auto count = (first & 0x07) + MINIMUM_REPEAT;
					INPUT_T tmp_int = 0;
					while (nbytes-- > 0) {
						tmp_int |= ((INPUT_T)read_byte() << (nbytes * 8));
					}
					while (count-- > 0) {
						write_int(tmp_int);
					}
				} break;
				case HEADER_DIRECT: {
#ifdef DEBUG
if (cid == ERR_CHUNK && tid == ERR_THREAD) {
	printf("thread %u case direct >>>\n", tid);
}
#endif
					uint8_t encoded_fbw = (first >> 1) & 0x1f;
					uint8_t fbw = get_decoded_bit_width(encoded_fbw);
					uint8_t len = ((static_cast<uint16_t>(first & 0x01) << 8) | read_byte()) + 1;
					uint8_t bits_left = 0 /* bits left over from unused bits of last byte */, curr_byte = 0;
					while (len-- > 0) {
						UINPUT_T result = 0;
						uint8_t bits_to_read = fbw;
						while (bits_to_read > bits_left) {
							result <<= bits_left;
							result |= curr_byte & ((1 << bits_left) - 1);
							bits_to_read -= bits_left;
							curr_byte = read_byte();
							bits_left = 8;
						}

						if (bits_to_read > 0) {
							result <<= bits_to_read;
							bits_left -= bits_to_read;
							result |= (curr_byte >> bits_left) & ((1 << bits_to_read) - 1);
						}

						write_int(static_cast<INPUT_T>(result));
					}
				} break;
				case HEADER_DELTA: {
#ifdef DEBUG
if (cid == ERR_CHUNK && tid == ERR_THREAD) {
	printf("thread %u case delta >>>\n", tid);
}
#endif
					uint8_t encoded_fbw = (first >> 1) & 0x1f;
					uint8_t fbw = get_decoded_bit_width(encoded_fbw);
					uint8_t len = ((static_cast<uint16_t>(first & 0x01) << 8) | read_byte()) + 1;

					INPUT_T base_val = static_cast<INPUT_T>(read_uvarint());
        			INPUT_T base_delta = static_cast<INPUT_T>(read_svarint());
					write_int(base_val);
					base_val += base_delta;
					write_int(base_val);

					len -= 2;
					int multiplier = (base_delta > 0 ? 1 : -1);
					if (encoded_fbw != 0) {
						uint8_t bits_left = 0, curr_byte = 0;
						while (len-- > 0) {
							UINPUT_T result = 0;
							uint8_t bits_to_read = fbw;
							while (bits_to_read > bits_left) {
								result <<= bits_left;
								result |= curr_byte & ((1 << bits_left) - 1);
								bits_to_read -= bits_left;
								curr_byte = read_byte();
								bits_left = 8;
							}

							if (bits_to_read > 0) {
								result <<= bits_to_read;
								bits_left -= bits_to_read;
								result |= (curr_byte >> bits_left) & ((1 << bits_to_read) - 1);
							}

							INPUT_T dlt = static_cast<INPUT_T>(result) * multiplier;
							base_val += dlt; 
							write_int(base_val);
						}
					} else {
						while (len-- > 0) {
							base_val += base_delta;
							write_int(base_val);
						}
					}
				} break;	
				case HEADER_PACTED_BASE: {
					uint8_t encoded_fbw = (first >> 1) & 0x1f;
					uint8_t fbw = get_decoded_bit_width(encoded_fbw);
					uint8_t len = ((static_cast<uint16_t>(first & 0x01) << 8) | read_byte()) + 1;

					uint8_t third = read_byte();
        			uint8_t forth = read_byte();

					uint8_t bw = ((third >> 5) & 0x07) + 1;
					uint8_t pw = get_decoded_bit_width(third & 0x1f);
					uint8_t pgw = ((forth >> 5) & 0x07) + 1;
					uint8_t pll = forth & 0x1f;

					uint32_t patch_mask = (static_cast<uint32_t>(1) << pw) - 1;

					uint32_t base_out_ptr = out_ptr;

					int64_t base_val = 0 ;
					while (bw-- > 0) {
						base_val |= ((INPUT_T)read_byte() << (bw * 8));
					}
#ifdef DEBUG
if (cid == ERR_CHUNK && tid == ERR_THREAD) {
	printf("thread %u case patched base with base val %ld\n", tid, base_val);
}
#endif
					uint8_t bits_left = 0 /* bits left over from unused bits of last byte */, curr_byte = 0;
					while (len-- > 0) {
						uint64_t result = 0;
						uint8_t bits_to_read = fbw;
						while (bits_to_read > bits_left) {
							result <<= bits_left;
							result |= curr_byte & ((1 << bits_left) - 1);
							bits_to_read -= bits_left;
							curr_byte = read_byte();
							bits_left = 8;
						}

						if (bits_to_read > 0) {
							result <<= bits_to_read;
							bits_left -= bits_to_read;
							result |= (curr_byte >> bits_left) & ((1 << bits_to_read) - 1);
						}
						
						write_int(static_cast<INPUT_T>(result) + base_val);
					}

					bits_left = 0, curr_byte = 0;
					uint8_t cfb = get_closest_bit(pw + pgw);
					int patch_gap = 0;
					while (pll-- > 0) {
						uint64_t result = 0;
						uint8_t bits_to_read = cfb;
						while (bits_to_read > bits_left) {
							result <<= bits_left;
							result |= curr_byte & ((1 << bits_left) - 1);
							bits_to_read -= bits_left;
							curr_byte = read_byte();
							bits_left = 8;
						}
						
						if (bits_to_read > 0) {
							result <<= bits_to_read;
							bits_left -= bits_to_read;
							result |= (curr_byte >> bits_left) & ((1 << bits_to_read) - 1);
						}
						patch_gap += (result >> pw);
#ifdef DEBUG
if (cid == ERR_CHUNK && tid == ERR_THREAD) {
	printf("thread %u patch gap %d\n", tid, patch_gap);
}
#endif
						uint32_t direct_out_ptr = base_out_ptr + patch_gap;
						INPUT_T *pb_ptr = nullptr;
						if (READ_UNIT < 4) {
							pb_ptr = &base_out[(direct_out_ptr / READ_UNIT) * BLK_SIZE * READ_UNIT + (direct_out_ptr % READ_UNIT) + tid * READ_UNIT];
						} else {
							// if (out_ptr - direct_out_ptr >= WRITE_VEC_SIZE || out_buffer_ptr == 0) {
							if (out_ptr / WRITE_VEC_SIZE - direct_out_ptr / WRITE_VEC_SIZE > 1) {
								pb_ptr = &base_out[(direct_out_ptr / READ_UNIT) * BLK_SIZE * READ_UNIT + (direct_out_ptr % READ_UNIT) + tid * READ_UNIT];
							} else if (out_ptr / WRITE_VEC_SIZE - direct_out_ptr / WRITE_VEC_SIZE == 1) {
								if (out_buffer_ptr == 0) {
									pb_ptr = &base_out[(direct_out_ptr / READ_UNIT) * BLK_SIZE * READ_UNIT + (direct_out_ptr % READ_UNIT) + tid * READ_UNIT];
								} else {
									pb_ptr = &out_buffer[tid][direct_out_ptr % WRITE_VEC_SIZE];
								}
							} else {
								pb_ptr = &out_buffer[tid][direct_out_ptr % WRITE_VEC_SIZE];
							}
						}
						*pb_ptr -= base_val;
						*pb_ptr |= (static_cast<INPUT_T>(result & patch_mask) << fbw);
						*pb_ptr += base_val;
					}
				} break;
				}
			}

			if (READ_UNIT >= 4 && out_buffer_ptr > 0) {
				deque_int();
			}
		}
    }
};

#endif