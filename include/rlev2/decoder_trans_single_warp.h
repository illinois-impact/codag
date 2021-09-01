#ifndef _RLEV2_DECODER_SINGLE_WARP_H_
#define _RLEV2_DECODER_SINGLE_WARP_H_

#include "utils.h"

namespace rlev2 {
	template <int READ_UNIT>
	__global__ void decompress_kernel_single_warp(const uint8_t* __restrict__ in, const uint64_t decomp_n_chunks, const blk_off_t* __restrict__ blk_off, const col_len_t* __restrict__ col_len, int64_t* __restrict__ out) {
        int tid = threadIdx.x;
        int cid = blockIdx.x;

        int wid = tid / 32;
        tid %= 32;

        uint64_t mychunk_size = col_len[(cid + wid * decomp_n_chunks) * BLK_SIZE + tid];
        uint64_t in_start_idx = blk_off[(cid + wid * decomp_n_chunks)];

        uint64_t out_start_idx = (cid + wid * decomp_n_chunks) * CHUNK_SIZE / sizeof(int64_t);

        uint32_t* in_4B = (uint32_t *)(&(in[in_start_idx]));
        uint32_t in_4B_off = 0;

        int64_t* out_8B = out + (out_start_idx + tid * READ_UNIT);
        // printf("out off %lu for thread %d\n", out_start_idx + tid * 1, tid);


        uint8_t shm_buffer[SINGLE_WARP_DECODE_BUFFER_COUNT];
        uint8_t *input_buffer = &shm_buffer[0];

        uint8_t curr_schm = 0;

        uint8_t input_buffer_head = 0;
        uint8_t input_buffer_tail = 0;
        uint8_t input_buffer_count = 0;

        uint8_t curr_fbw = 0, curr_fbw_left = 0;

        uint8_t first_byte;

        // for complicated encoding schemes, input_buffer may not have enough data
        // needs a structure to hold the info of last encoding scheme
        int curr_len = 0;

        int64_t curr_64; // current outint, might not be complete
        uint16_t bits_left = 0;
        uint8_t bits_left_over; // leftover from last byte since bit packing

        bool dal_read_base = false;
        int64_t base_val, base_delta, *base_out; // for delta encoding

        int bw, pw, pgw, pll, patch_gap, curr_pwb_left; // for patched base encoding

        uint64_t used_bytes = 0;

        int curr_write_offset = 0;
        int64_t out_buffer[WRITE_VEC_SIZE];
        uint8_t out_buffer_ptr = 0;
        uint8_t out_counter = 0;

        auto read_byte = [&]() {
                auto ret = input_buffer[input_buffer_head];
                input_buffer_count -= 1;
                used_bytes += 1;
                input_buffer_head = (input_buffer_head + 1) % SINGLE_WARP_DECODE_BUFFER_COUNT;
                return ret;
        };

        auto deque_int = [&]() {
                *reinterpret_cast<longlong4*>(out_8B + out_counter) = *reinterpret_cast<longlong4*>(out_buffer);
                out_counter += WRITE_VEC_SIZE;
                if (out_counter == READ_UNIT) {
                        out_counter = 0;
                        out_8B += BLK_SIZE * READ_UNIT;
                }
                out_buffer_ptr = 0;
        };

        auto write_int = [&](int64_t i) {
                if (out_buffer_ptr == WRITE_VEC_SIZE) {
                        deque_int();
                }

                out_buffer[out_buffer_ptr++] = i;

                curr_len --;
                curr_64 = 0;
                curr_fbw_left = curr_fbw;

        //      *(out_8B + curr_write_offset) = i;
//     curr_write_offset ++;
//     if (curr_write_offset == READ_UNIT) {
//         curr_write_offset = 0;
//         out_8B += BLK_SIZE * READ_UNIT;
//     }

        //      curr_len --;
        //      curr_64 = 0;
        //      curr_fbw_left = curr_fbw;
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

        const uint32_t t_read_mask = (0xffffffff >> (32 - tid));

        while (used_bytes < mychunk_size) {

            auto mask = __activemask();
                bool read;
                #pragma unroll
                for (int read_iter=0; read_iter<2; ++read_iter) {

                        read = used_bytes + input_buffer_count < mychunk_size;
                        uint32_t read_sync = __ballot_sync(mask, read);
                        if (read) {
                                *(uint32_t *)(&(input_buffer[input_buffer_tail])) = in_4B[in_4B_off + __popc(read_sync & t_read_mask)];
                                input_buffer_tail = (input_buffer_tail + 4) % SINGLE_WARP_DECODE_BUFFER_COUNT;
                                input_buffer_count += 4;
                                in_4B_off += __popc(read_sync);
                        }
                        __syncwarp(mask);
                }


                if (curr_schm == 0) {
                        first_byte = read_byte();
                        curr_schm = first_byte & HEADER_MASK;
                        curr_fbw = get_decoded_bit_width((first_byte >> 1) & 0x1f);
                        curr_fbw_left = curr_fbw;

                        if (curr_schm != HEADER_SHORT_REPEAT) {
                                curr_len = (((first_byte & 0x01) << 8) | read_byte()) + 1;
                                bits_left = 0; bits_left_over = 0; curr_64 = 0;

                                dal_read_base = false;

                                if (curr_schm == HEADER_PACTED_BASE) {
                                        auto third = read_byte();
                                        auto forth = read_byte();

                                        bw = ((third >> 5) & 0x07) + 1;
                                        pw = get_decoded_bit_width(third & 0x1f);
                                        pgw = ((forth >> 5) & 0x07) + 1;
                                        pll = forth & 0x1f;
                                        patch_gap = 0;

                                        curr_pwb_left = get_closest_bit(pw + pgw);
                                }
                        }
                }

                switch(curr_schm) {
                case HEADER_SHORT_REPEAT: {
                        auto num_bytes = ((first_byte >> 3) & 0x07) + 1;
                        if (num_bytes <= input_buffer_count) {
                                int64_t tmp_int = 0;
                                while (num_bytes-- > 0) {
                                        tmp_int |= ((int64_t)read_byte() << (num_bytes * 8));
                                }
                                auto cnt = (first_byte & 0x07) + MINIMUM_REPEAT;
                                while (cnt-- > 0) {
                                        // *(out_8B + curr_write_offset) = tmp_int;
                // curr_write_offset ++;
                // if (curr_write_offset == READ_UNIT) {
                //     curr_write_offset = 0;
                                        //     out_8B += BLK_SIZE * READ_UNIT;
                // }
                                        write_int(tmp_int);
                                }
                                curr_schm = 0;
                        }
                }       break;
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

                        curr_schm = 0;
                }       break;
                case HEADER_DELTA: {
                        if (!dal_read_base) {
                                if (read && input_buffer_count < 64 + curr_fbw) break;
                                dal_read_base = true;

                                base_val = read_uvarint();
                                base_delta = read_svarint();
                                write_int(base_val);
                                base_val += base_delta;
                                write_int(base_val);
                        }

                        if (((first_byte >> 1) & 0x1f) != 0) {
                                // var delta
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
                                                base_val += curr_64; //TODO: THIS IS NOT ALWAYS +.
                                                write_int(base_val);
                                        }
                                }
                        } else {
                                // fixed delta encoding
                                while (curr_len > 0) {
                                        base_val += base_delta;
                                        write_int(base_val);
                                }
                        }
                        curr_schm = 0;
                }       break;
                case HEADER_PACTED_BASE: {
                        if (!dal_read_base) {
                                if (input_buffer_count < bw) break;
                                dal_read_base = true;

                                base_val = 0;
                                auto fbw = bw;
                                while (fbw-- > 0) {
                                        base_val |= (read_byte() << (fbw * 8));
                                }
                                base_out = out_8B;
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

                        auto patch_mask = (static_cast<uint32_t>(1) << pw) - 1;
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
                                        base_out[(patch_gap / READ_UNIT) * BLK_SIZE + (patch_gap % READ_UNIT)] |= static_cast<int64_t>(curr_64 & patch_mask) << curr_fbw;

                                        pll --;
                                        curr_64 = 0;
                                        curr_pwb_left = get_closest_bit(pw + pgw);
                                }
                        }
                        curr_schm = 0;
                }       break;
                default: {
                        curr_schm = 0;
                } break;
                }
                main_loop:
                __syncwarp(mask);
}
        if (out_buffer_ptr > 0) {
                deque_int();
        }

}

	
    template <int READ_UNIT, int DECOMP_BLK_SIZE=32>
    void decompress_single_warp(const uint8_t* __restrict__ in, const uint64_t n_chunks, const blk_off_t* __restrict__ blk_off, const col_len_t* __restrict__ col_len, int64_t* __restrict__ out) {
        assert(DECOMP_BLK_SIZE % 32 == 0);

        uint64_t n_threads = n_chunks * 32;
        uint64_t decomp_n_chunks = (n_threads - 1) / DECOMP_BLK_SIZE + 1;

        decompress_kernel_single_warp<READ_UNIT><<<decomp_n_chunks, DECOMP_BLK_SIZE>>>(in, decomp_n_chunks, blk_off, col_len, out);
    }

}
#endif