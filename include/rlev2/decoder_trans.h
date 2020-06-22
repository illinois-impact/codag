#ifndef _RLEV2_DECODER_TRANPOSE_H_
#define _RLEV2_DECODER_TRANPOSE_H_

#include "utils.h"
#include <stdio.h>

namespace rlev2 {
	__host__ __device__ int readVulong(uint8_t* in, int64_t& out) {
        out = 0;
        int offset = 0;
		int proceed = 0;
		uint8_t b = 0;
        do {
            b = *(in ++);
			proceed ++;
            out |= (0x7f & b) << offset;
            offset += 7;
        } while (b >= 0x80);
		return proceed;
    }

	__host__ __device__ inline int64_t unZigZag(uint64_t value) {
        return value >> 1 ^ -(value & 1);
    }

	__host__ __device__ int readVslong(uint8_t* in, int64_t& out) {
		int ret = readVulong(in, out);
		unZigZag(out);
        return ret;
    }

    void decompress_func_new(const uint8_t* const in, int64_t* out, 
                        const uint64_t in_n_bytes, const uint64_t out_n_bytes, 
                        const uint64_t out_chunk_size, const uint64_t n_chunks, 
                        const uint64_t* const blk_off, const uint64_t* const col_len, const uint8_t* const col_map) {
		// int tid = threadIdx.x;
		int tid = 0;
		// int chunk_idx = blockIdx.x;
		int chunk_idx = 0;
		uint64_t used_bytes = 0;
		// uint64_t mychunk_size = col_len[blockDim.x * chunk_idx + tid];
		uint64_t mychunk_size = col_len[0 * chunk_idx + tid];
		uint64_t in_start_idx = blk_off[chunk_idx];

		uint8_t header_byte = 0;
		uint8_t block = 0;

		uint64_t out_bytes = 0;
		uint64_t out_start_idx = chunk_idx * CHUNK_SIZE;

		uint8_t compress_counter = 0;
		// int col_idx = col_map[blockDim.x * chunk_idx + tid];
		int col_idx = col_map[0 * chunk_idx + tid];

		uint8_t v = 0;
		uint32_t* in_4B = (uint32_t *)(&(in[in_start_idx]));
		uint32_t in_4B_off = 0;


		int data_counter = 0;

		int64_t out_buffer = 0;
		uint8_t* out_buffer_8 = (uint8_t*) &out_buffer;
		uint8_t out_buffer_tail = 0;
		uint64_t out_off = 0;

		int64_t* out_8B = (int64_t*)(&(out[out_start_idx + col_idx*4]));

		uint8_t input_buffer[INPUT_BUFFER_SIZE];
		uint8_t input_buffer_head = 0;
		uint8_t input_buffer_tail = 0;
		uint8_t input_buffer_count = 0;
		// uint8_t input_buffer_read_count = 0;

		bool stall_flag = false;
		uint8_t header_off = 0;
		bool type = 0;
		uint32_t type_0_byte = 0;
		uint64_t type_0_v = 0;
		uint64_t read_count = 0;

		bool read_first = false, read_second = false, read_third = false, read_forth = false;
		uint8_t first_byte, third_byte, forth_byte;

		// for complicated encoding schemes, input_buffer may not have enough data
		// needs a structure to hold the info of last encoding scheme
		uint16_t curr_len; 

		int64_t curr_64; // current outint, might not be complete
		uint16_t bits_left_to_read, bits_left;
		uint8_t bits_left_over; // leftover from last byte since bit packing

		bool dal_read_base = false;


#define proceed(x) { \
	input_buffer_count -= x; \
	used_bytes += x; \
	input_buffer_head = (input_buffer_head + 1) % INPUT_BUFFER_SIZE; \
}

#define write_varint(i) { \
	*(out_8B ++) = i; \
}

		static auto read_uvarint = [&input_buffer, &input_buffer_head, &input_buffer_count, &used_bytes]() {
			int64_t out = 0;
			int offset = 0;
			uint8_t b = 0;
			do {
				b = input_buffer[input_buffer_head];
				proceed(1);
				out |= (0x7f & b) << offset;
				offset += 7;
			} while (b >= 0x80);
			return out;
		};

		static auto read_svarint = [&input_buffer, &input_buffer_head, &input_buffer_count, &used_bytes]() {
			auto ret = static_cast<int64_t>(read_uvarint());
			return ret >> 1 ^ -(ret & 1);
		};


		int64_t base_val, base_delta; // for delta encoding

		// printf("chunk size: %lu\n", mychunk_size);

        while (used_bytes < mychunk_size) {
            // unsigned mask = __activemask();
            bool read = read_count < mychunk_size;
			// int res = __popc(__ballot_sync(mask, (read)));
            if (read && input_buffer_count + 4 <= INPUT_BUFFER_SIZE) {
                uint32_t* input_buffer_4B = (uint32_t *)(&(input_buffer[input_buffer_tail]));
				input_buffer_4B[0] = in_4B[in_4B_off + tid];

				input_buffer_tail = (input_buffer_tail + 4) % INPUT_BUFFER_SIZE;
				input_buffer_count += 4;
				read_count += 4;
		
				in_4B_off += 1;

            } 

			if (!read_first) {
				first_byte = input_buffer[input_buffer_head];
				// printf("read first byte %x\n", first_byte);
				proceed(1);
				read_first = true;
			}

			switch(first_byte & 0xC0) {
			case HEADER_SHORT_REPEAT: {
				// printf("cas	e short repeat\n");
				auto num_bytes = ((first_byte >> 3) & 0x07) + 1;
				if (num_bytes <= input_buffer_count) { 
				//TODO: this is guaranteed since for short repeat encoded bytes are small enough
					curr_64 = 0;
					while (num_bytes-- > 0) {
						curr_64 |= (input_buffer[input_buffer_head] << (num_bytes * 8));
						proceed(1);
					}
					auto cnt = (first_byte & 0x07) + MINIMUM_REPEAT;
					// printf("write %ld %u times\n", curr_64, cnt);
					while (cnt-- > 0) {
						write_varint(curr_64);
					}
				}
				read_first = false;
			}	break;
			case HEADER_DIRECT: {
				// printf("case direct\n");
        		uint8_t curr_fbw = get_decoded_bit_width((first_byte >> 1) & 0x1f);
				if (!read_second) {
					read_second = true;
					curr_len = ((static_cast<uint16_t>(first_byte & 0x01) << 8) | input_buffer[input_buffer_head]) + 1;
					proceed(1);
					bits_left = 0;
					bits_left_over = 0;
					curr_64 = 0;
				}
				while (input_buffer_count && curr_len > 0) {
					bits_left_to_read = curr_fbw;
					while (input_buffer_count && bits_left_to_read > bits_left) {
						curr_64 <<= bits_left;
						curr_64 |= bits_left_over & ((1 << bits_left) - 1);
						bits_left_to_read -= bits_left;
						bits_left_over = input_buffer[input_buffer_head];
						proceed(1);
						bits_left = 8;
					}

					if (bits_left_to_read <= bits_left) {
					// if not curr_64 is imcomplete
						if (bits_left_to_read > 0) {
							curr_64 <<= bits_left_to_read;
							bits_left -= bits_left_to_read;
							curr_64 |= (bits_left_over >> bits_left) & ((1 << bits_left_to_read) - 1);
						}

						write_varint(curr_64);
						curr_len --;
						curr_64 = 0;
					}
				}
				if (curr_len == 0) {
					read_first = false; read_second = false;
				}
			}	break;
			case HEADER_DELTA: {
        		uint8_t curr_fbw = get_decoded_bit_width((first_byte >> 1) & 0x1f);
				printf("bits needed %u\n", curr_fbw);
				if (!read_second) {
					read_second = true;
					curr_len = ((static_cast<uint16_t>(first_byte & 0x01) << 8) | input_buffer[input_buffer_head]) + 1;
					proceed(1);
					bits_left = 0;
					bits_left_over = 0;
					curr_64 = 0;
				}

				// TODO: double check whether curr_fbw represent base length or delta length 
				if (!dal_read_base && input_buffer_count >= 2 * curr_fbw / 8) {
					base_val = read_uvarint();
					base_delta = read_svarint();
					
					write_varint(base_val);
					base_val += base_delta;
					write_varint(base_val);

					curr_len -= 2;
					
					// proceed(readVulong(input_buffer, base_val));
					// proceed(readVslong(input_buffer, base_delta));
					
					// printf("read Long %ld\n", base_val);
					// printf("read Long %ld\n", base_delta);
					dal_read_base = true;
				}

				if (dal_read_base) {
					if (curr_fbw != 0) {
						while (input_buffer_count && curr_len > 0) {
							bits_left_to_read = curr_fbw;
							while (input_buffer_count && bits_left_to_read > bits_left) {
								curr_64 <<= bits_left;
								curr_64 |= bits_left_over & ((1 << bits_left) - 1);
								bits_left_to_read -= bits_left;
								bits_left_over = input_buffer[input_buffer_head];
								proceed(1);
								bits_left = 8;
							}

							if (bits_left_to_read <= bits_left) {
							// if not curr_64 is imcomplete
								if (bits_left_to_read > 0) {
									curr_64 <<= bits_left_to_read;
									bits_left -= bits_left_to_read;
									curr_64 |= (bits_left_over >> bits_left) & ((1 << bits_left_to_read) - 1);
								}

								write_varint(curr_64);
								curr_len --;
								curr_64 = 0;
							}
						}
						
					} else {
						while (curr_len-- > 0) {
							base_val += base_delta;
							write_varint(base_val);
						}
					}
					if (curr_len == 0) {
						read_first = false; read_second = false;
					}
				}

			}	break;
			case HEADER_PACTED_BASE: {

			}	break;
			}
        }
#undef proceed
#undef write_varint

    }
}

#endif