#ifndef _RLEV2_DECODER_TRANPOSE_H_
#define _RLEV2_DECODER_TRANPOSE_H_

#include "utils.h"
#include <stdio.h>

#define DEBUG

namespace rlev2 {
	__global__
	void decompress_func_new1(const uint8_t* in, const uint64_t in_n_bytes, 
						const uint64_t n_chunks, 
                        const blk_off_t* blk_off, const col_len_t* col_len,
						int64_t* out) {
		uint32_t tid = threadIdx.x;
		uint32_t cid = blockIdx.x;
		uint64_t used_bytes = 0;
		
		uint64_t mychunk_size = col_len[cid * BLK_SIZE + tid];
		uint64_t in_start_idx = blk_off[cid];

		uint64_t out_start_idx = cid * CHUNK_SIZE / sizeof(int64_t);

		// if (cid == 0 && tid == 0) printf("thread 0 start from %lu\n", in_start_idx);

		uint32_t* in_4B = (uint32_t *)(&(in[in_start_idx]));
		uint32_t in_4B_off = 0;

		int64_t* out_8B = out + (out_start_idx + tid * 1); //TODO: should be READ_UNIT here
		// printf("out off %lu for thread %d\n", out_start_idx + tid * 1, tid);

		uint8_t input_buffer[INPUT_BUFFER_SIZE];
		uint8_t input_buffer_head = 0;
		uint8_t input_buffer_tail = 0;
		uint8_t input_buffer_count = 0;
		// uint8_t input_buffer_read_count = 0;

		uint64_t read_count = 0;

		bool read_first = false, read_second = false;
		uint8_t first_byte;

		// for complicated encoding schemes, input_buffer may not have enough data
		// needs a structure to hold the info of last encoding scheme
		int curr_len = 0; 

		int64_t curr_64; // current outint, might not be complete
		uint16_t bits_left = 0;
		uint8_t bits_left_over; // leftover from last byte since bit packing

		bool dal_read_base = false;
		int64_t base_val, base_delta, *base_out; // for delta encoding

		uint8_t bw, pw, pgw, pll, patch_gap; // for patched base encoding

	// printf("thread %d write %ld\n", tid, i); \

		auto read_byte = [&]() {
			auto ret = input_buffer[input_buffer_head];
			input_buffer_count -= 1;
			used_bytes += 1;
			input_buffer_head = (input_buffer_head + 1) % INPUT_BUFFER_SIZE;
			return ret;
		};

		auto write_int = [&](int64_t i) {
// if (cid == 0 && tid == 0) {
// printf("chunk %u thread 0 write %ld(%lu)\n", cid, i, out_8B - out);
// }
			*out_8B = i; 
			out_8B += BLK_SIZE; 

		};

		// by default, lambda captures what was referecnes in lambda
		auto read_uvarint = [&]() {
			uint64_t out = 0;
			int offset = 0;
			uint8_t b = 0;
			do {
				b = read_byte();
				out |= (0x7f & b) << offset;
				offset += 7;
			} while (b >= 0x80);
			return out;
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

		auto read_longs = [&](uint8_t fbw) {
			// int max_iter = 10, iter = 0;
			while (curr_len > 0 && (input_buffer_count || bits_left)) {
// if (tid == 0) printf("(currlen)input_count(bits_left) (%d)%u(%u)\n", curr_len, input_buffer_count, bits_left);
// 				if (++iter > max_iter) {
// 					printf("PATCHED BASE: this line should not acuur\n");
// 					break;
// 				}
				int bits_left_to_read = fbw;
				while (input_buffer_count && bits_left_to_read > bits_left) {
					curr_64 <<= bits_left;
					curr_64 |= bits_left_over & ((1 << bits_left) - 1);
					bits_left_to_read -= bits_left;
					bits_left_over = read_byte();
					bits_left = 8;
					
				}

				if (bits_left_to_read <= bits_left) {
				// if not curr_64 is imcomplete
					if (bits_left_to_read > 0) {
						curr_64 <<= bits_left_to_read;
						bits_left -= bits_left_to_read;
						curr_64 |= (bits_left_over >> bits_left) & ((1 << bits_left_to_read) - 1);
					}

					write_int(curr_64);
					curr_len --;
					curr_64 = 0;
				} else {
					if (input_buffer_count == 0) break;
				}
			}
		};

// if (tid == ERR_THREAD) printf("thread %d chunk size: %lu\n", ERR_THREAD, mychunk_size);
		// bits_left is for 3 other encoding schemes
        while (used_bytes < mychunk_size || curr_len > 0) {
			// if (cid == 0 && tid == 0) {
			// 	printf("used %u(%u) bytes\n", used_bytes, mychunk_size);
			// }
			// printf("loop\n");
            auto mask = __activemask();
            bool read = read_count < mychunk_size;
			auto res = __popc(__ballot_sync(mask, (read)));

			int left_active = 0;
			for (int i=0; i<tid; ++i) {
				if (read_count < col_len[cid * BLK_SIZE + i]) {
					++left_active;
				}
			}
// if (cid == 0 && tid == ERR_THREAD) {
// printf("read active thread at %luth iter: %u(%d)\n",read_count / 4, res, left_active);
// }
            if (read && input_buffer_count + 4 <= INPUT_BUFFER_SIZE) {
                uint32_t* input_buffer_4B = (uint32_t *)(&(input_buffer[input_buffer_tail]));
				input_buffer_4B[0] = in_4B[in_4B_off + left_active];  

if (cid == 0 && tid == ERR_THREAD) {
	printf("chunk %d thread %d reads at pos(%u) %x%x%x%x\n", cid, tid, (in_4B_off + left_active) * 4,
		input_buffer[input_buffer_head], 
		input_buffer[input_buffer_head+1], 
		input_buffer[input_buffer_head+2], 
		input_buffer[input_buffer_head+3]);
}
				input_buffer_tail = (input_buffer_tail + 4) % INPUT_BUFFER_SIZE;
				input_buffer_count += 4;
				read_count += 4;
		
				in_4B_off += res; 
            } 

			if (!read_first) {
				first_byte = read_byte();
				// printf("read first byte %x\n", first_byte);
				read_first = true;
				read_second = false;
			}

			switch(first_byte & 0xC0) {
			case HEADER_SHORT_REPEAT: {
				// printf("case short repeat\n");
				auto num_bytes = ((first_byte >> 3) & 0x07) + 1;
				if (num_bytes <= input_buffer_count) { 
				//TODO: this is guaranteed since for short repeat encoded bytes are small enough
					curr_64 = 0;
					while (num_bytes-- > 0) {
						curr_64 |= (read_byte() << (num_bytes * 8));
					}
					auto cnt = (first_byte & 0x07) + MINIMUM_REPEAT;
					// printf("write %ld %u times\n", curr_64, cnt);
					while (cnt-- > 0) {
						write_int(curr_64);
					}
				}
				read_first = false;
			}	break;
			case HEADER_DIRECT: {
				// printf("======> case direct\n");
        		uint8_t curr_fbw = get_decoded_bit_width((first_byte >> 1) & 0x1f);
				if (!read_second) {
					read_second = true;
					curr_len = (((first_byte & 0x01) << 8) | read_byte()) + 1;
					bits_left = 0;
					bits_left_over = 0;
					curr_64 = 0;
#ifdef DEBUG
if (tid + cid == 0) printf("HEADER_DIRECT: %d len of ints\n", curr_len);
#endif
				}
				
				read_longs(curr_fbw);
				if (curr_len <= 0) {
#ifdef DEBUG
if (curr_len < 0) printf("HEADER_DIRECT: this line should not occured\n");
#endif
					read_first = false; read_second = false;
				}
			}	break;
			case HEADER_DELTA: {
				// printf("======> case delta\n");
        		uint8_t curr_fbw = get_decoded_bit_width((first_byte >> 1) & 0x1f);
				if (!read_second) {
					read_second = true;
					curr_len = (((first_byte & 0x01) << 8) | read_byte()) + 1;
					bits_left = 0;
					bits_left_over = 0;
					curr_64 = 0;
					dal_read_base = false;
				}

				//curr_fbw is delta bits.
				if (!dal_read_base && (input_buffer_count >= 8 + curr_fbw || !read)) {//TODO: should be min(fbw, 8)
// #ifdef DEBUG
// if (tid == 0) {
// printf("left to read base %d\n", input_buffer_count);
// }
// #endif
					base_val = read_uvarint();
// #ifdef DEBUG
// if (tid == 0) {
// printf("left to read delta %d\n", input_buffer_count);
// }
// #endif
					base_delta = read_svarint();
					
					write_int(base_val);
					base_val += base_delta;
					write_int(base_val);

// #ifdef DEBUG
// if (tid == 0) {
// printf("thread 0 base: %ld\n", base_val);
// printf("thread 0 delta: %ld\n", base_delta);
// }
// #endif

					base_out = out_8B;

					curr_len -= 2;
					dal_read_base = true;
				}

				if (dal_read_base) {
					if (((first_byte >> 1) & 0x1f) != 0) {
						// var delta
						read_longs(curr_fbw);
						
					} else {
						// fixed delta encoding
						while (curr_len > 0) {
							base_val += base_delta;
							write_int(base_val);
							curr_len --;
						}
					}

					if (curr_len <= 0) {
#ifdef DEBUG
if (curr_len < 0) printf("HEADER_DELTA: this line should not occured\n");
#endif
						if (((first_byte >> 1) & 0x1f) != 0) {
							if (base_delta > 0) {
								while (base_out < out_8B) {
									base_val = *base_out += base_val;
									base_out += BLK_SIZE;
								}
							} else {
								while (base_out < out_8B) {
									base_val = *base_out = base_val - *base_out;
									base_out += BLK_SIZE;
								}
							}
						}
						
						read_first = false; read_second = false;
					}
				}
			}	break;
			case HEADER_PACTED_BASE: {
// if (tid == 0) printf("======> case patched baes\n");

				//TODO Try to guarantee there are at least 4 btyes to read (all headers)
				uint8_t curr_fbw = get_decoded_bit_width((first_byte >> 1) & 0x1f);
				if (!read_second) {
					if (input_buffer_count < 3) break;
				// Here for patched base, read_second includes third and forth header byte
					curr_len = (((first_byte & 0x01) << 8) | read_byte()) + 1;
					auto third = read_byte();
        			auto forth = read_byte();

					bw = ((third >> 5) & 0x07) + 1;
					pw = get_decoded_bit_width(third & 0x1f);
					pgw = ((forth >> 5) & 0x07) + 1;
					pll = forth & 0x1f;

					bits_left = 0;
					bits_left_over = 0;
					curr_64 = 0;
					patch_gap = 0;
					dal_read_base = false;

					read_second = true;
				}
				
				if (!read_second) break;

				if (!dal_read_base && input_buffer_count >= bw) {
					base_val = read_long(bw);
					base_out = out_8B;
					dal_read_base = true;
				}

				if (!dal_read_base) break;

				if (curr_len > 0) {
					read_longs(curr_fbw);
// if (tid == 0)  printf("curr_len after read longs: %d(%u)\n", curr_len,curr_fbw);
				} else {
					// process patched list
// if (tid == 0) printf("======> process patch base	\n");
					auto cfb = get_closest_bit(pw + pgw);
					auto patch_mask = (static_cast<uint16_t>(1) << pw) - 1;
					while (pll > 0 && (input_buffer_count || bits_left)) {
						auto bits_left_to_read = cfb;
						while (input_buffer_count && bits_left_to_read > bits_left) {
							curr_64 <<= bits_left;
							curr_64 |= bits_left_over & ((1 << bits_left) - 1);
							bits_left_to_read -= bits_left;
							bits_left_over = read_byte();
							bits_left = 8;
						}

						if (bits_left_to_read <= bits_left) {
						// if not curr_64 is imcomplete
							if (bits_left_to_read > 0) {
								curr_64 <<= bits_left_to_read;
								bits_left -= bits_left_to_read;
								curr_64 |= (bits_left_over >> bits_left) & ((1 << bits_left_to_read) - 1);
							}

							// write_int(curr_64);
							patch_gap += curr_64 >> pw;
							base_out[patch_gap * BLK_SIZE] |= static_cast<int64_t>(curr_64 & patch_mask) << curr_fbw;


							pll --;
							curr_64 = 0;
						} else {
							if (input_buffer_count == 0) break;
						}
					}
					if (pll <= 0) {
// if (pll < 0) printf("PACTHED BASE: this line should not appear\n");
						while (base_out < out_8B) {
							*base_out += base_val;
							base_out += BLK_SIZE;
						}
						read_first = false; read_second = false;
					}
				}
			}	break;
			}

			// if (cid + tid == 0) {
			// 	printf("used bytes: %lu\n", used_bytes);
			// 	printf("mychunk_size: %lu\n", mychunk_size);
			// 	printf("curr_len: %d\n", curr_len);
			// }
			// if (iter++ > 10) {
			// 	printf("iter shold not go over 10\n");
			// 	// if (cid == 0 && tid == 0) {
			// 	// 	printf("used bytes: %lu\n", used_bytes);
			// 	// 	printf("mychunk_size: %lu\n", mychunk_size);
			// 	// 	printf("curr_len: %d\n", curr_len);
			// 	// }
			// 	break;

			// }
			// if (read_count >= 16) break; //BREAK AFTER ONE ITER [DEL]
			__syncwarp(mask);
        }
#undef write_int
		

    }

	__host__
	void decompress_gpu(const uint8_t *in, const uint64_t in_n_bytes, const uint64_t n_chunks,
			blk_off_t *blk_off, col_len_t *col_len,
			int64_t *&out, uint64_t &out_n_bytes) {
		initialize_bit_maps();
		printf("======> decompress kernerl\n");
		// printf("======> n_chunks: %u\n", n_chunks);
		// printf("======> in_n_bytes: %u\n", in_n_bytes);
		// return;
		uint8_t *d_in;
		int64_t *d_out;
		blk_off_t *d_blk_off;
		col_len_t *d_col_len;

		auto exp_out_n_bytes = blk_off[n_chunks];
		out_n_bytes = exp_out_n_bytes;


		cuda_err_chk(cudaMalloc(&d_in, in_n_bytes));
		cuda_err_chk(cudaMalloc(&d_out, exp_out_n_bytes));
		cuda_err_chk(cudaMalloc(&d_blk_off, sizeof(blk_off_t) * n_chunks));
		cuda_err_chk(cudaMalloc(&d_col_len, sizeof(col_len_t) * n_chunks * BLK_SIZE));
			
		cuda_err_chk(cudaMemcpy(d_in, in, in_n_bytes, cudaMemcpyHostToDevice));
		cuda_err_chk(cudaMemcpy(d_blk_off, blk_off, sizeof(blk_off_t) * n_chunks, cudaMemcpyHostToDevice));
		cuda_err_chk(cudaMemcpy(d_col_len, col_len, sizeof(col_len_t) * n_chunks * BLK_SIZE, cudaMemcpyHostToDevice));

		
		decompress_func_new1<<<n_chunks, BLK_SIZE>>>(d_in, in_n_bytes, n_chunks,
					d_blk_off, d_col_len,
					d_out);
		cuda_err_chk(cudaDeviceSynchronize());


		out = new int64_t[exp_out_n_bytes / sizeof(int64_t)];
		cuda_err_chk(cudaMemcpy(out, d_out, exp_out_n_bytes, cudaMemcpyDeviceToHost));
		
		
		int p = 3840;
		for (int i=p; i<p+92; ++i) {
			printf("out[%d]: %ld\n", i, out[i]);
		}


		cuda_err_chk(cudaFree(d_in));
		cuda_err_chk(cudaFree(d_out));
		cuda_err_chk(cudaFree(d_blk_off));
		cuda_err_chk(cudaFree(d_col_len));
	}

}
#endif