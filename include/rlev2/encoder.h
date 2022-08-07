#ifndef _RLEV2_ENCODER_H_
#define _RLEV2_ENCODER_H_

#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include "utils.h"

namespace rlev2 {

    template<bool skip=false>
    struct encode_info {
        uint8_t *output;
        uint32_t potision = 0;
        
        INPUT_T deltas[MAX_LITERAL_SIZE];
        INPUT_T literals[MAX_LITERAL_SIZE];

        uint32_t num_literals = 0;
        uint32_t fix_runlen = 0, var_runlen = 0;
        
       
        

        uint8_t delta_bits; // is set when delta encoding is used

        bool is_fixed_delta;

        __host__ __device__
        inline void write_value(uint8_t val) {
                output[potision ++] = val;
        }

        int cid, tid;

        __host__ __device__
        encode_info() {
            for (int i=0; i<MAX_LITERAL_SIZE; ++i) {
                deltas[i] = 0;
            }
        }

    };

    typedef struct patch_blob {
        uint32_t patch_len;
        uint32_t patch_width;
        uint32_t patch_gap_width;
        uint8_t bits95p, bits100p; //should be reduced values' bits
        INPUT_T literal_min; //used for reducing other literals
        INPUT_T reduced_literals[MAX_LITERAL_SIZE];
        INPUT_T gap_patch_list[MAX_LITERAL_SIZE];
    } patch_blob;

    // Should guarantee first parameter is the larger one.
    __host__ __device__
    bool is_safe_subtract(const INPUT_T& larger, const INPUT_T& smaller) {
        // Original implementation causes overflow already while checking safety.
        if (smaller > 0) return true;
        return (larger > smaller - INT8_MIN);
    }

    __host__ __device__ void block_encode(const uint64_t, INPUT_T*, const uint64_t, uint8_t*, uint64_t*, uint64_t);
    __host__ __device__ void writeDirectValues(encode_info<>&);
    __host__ __device__ void writeDeltaValues(encode_info<>&);
    __host__ __device__ void writeShortRepeatValues(encode_info<>&);
    __host__ __device__ void writePatchedBasedValues(encode_info<>&, patch_blob&);

    // Only need 8 bit (255) to represent 64 bit varint
    __host__ __device__
    uint8_t find_closes_num_bits(INPUT_T value) {
        if (value < 0) {
            return get_closest_bit(64);
        }
        uint8_t count = 0;
        while (value) {
            value = value >> 1;
            ++ count;
        }
        return get_closest_bit(count);
    }

    __host__ __device__ 
    void write_aligned_ints(INPUT_T* in, uint32_t len, uint8_t bits, encode_info<>& info) {
        if (bits < 8 ) {
            // For safetyness, only bits should be written. 
            auto bitMask = static_cast<uint8_t>((1 << bits) - 1);
            uint32_t numHops = 8 / bits;
            uint32_t remainder = static_cast<uint32_t>(len % numHops);

            uint32_t endUnroll = len - remainder;
            for (uint32_t i=0; i < endUnroll; i+=numHops) {
                uint8_t toWrite = 0;
                for (uint32_t j = 0; j < numHops; ++j) {
                    toWrite |= static_cast<char>((in[i+j] & bitMask) << (8 - (j + 1) * bits));
                }
                info.write_value(toWrite);
            }

            if (remainder > 0) {
                uint32_t startShift = 8 - bits;
                uint8_t toWrite = 0;
                for (uint32_t i = endUnroll; i < len; ++i) {
                    toWrite |= static_cast<char>((in[i] & bitMask) << startShift);
                    startShift -= bits;
                }

                info.write_value(toWrite);
            }
        } else {
            int32_t bytes = bits / 8;
            for (uint32_t i=0; i<len; ++i) {
                for(int32_t j=bytes-1; j>=0; --j) {
                    // write int in big endianess
                    info.write_value(static_cast<uint8_t>((in[i] >> (j * 8)) & 0xff));
                }
            }
        }

    }

    __host__ __device__
    void write_unaligned_ints(INPUT_T* in, uint32_t len, uint8_t bits, encode_info<>& info) {
        uint32_t bitsLeft = 8;
        uint8_t current = 0;
        for(uint32_t i=0; i <len; i++) {
            INPUT_T value = in[i];
	    // printf("pb write val: %ld\n", value);
            uint32_t bitsToWrite = bits;

            while (bitsToWrite > bitsLeft) {
                // add the bits to the bottom of the current word
                current |= static_cast<uint8_t>(value >> (bitsToWrite - bitsLeft));
                // subtract out the bits we just added
                bitsToWrite -= bitsLeft;
                // zero out the bits above bitsToWrite
                value &= (static_cast<UINPUT_T>(1) << bitsToWrite) - 1;

                info.write_value(current);
                // printf("write byte %x(%u)\n", info.output[info.potision - 1], bits);
                current = 0;
                bitsLeft = 8;
            }
            bitsLeft -= bitsToWrite;
            current |= static_cast<uint8_t>(value << bitsLeft);
            if (bitsLeft == 0) {
                info.write_value(current);
                // printf("write byte %x\n", info.output[info.potision - 1]);
                current = 0;
                bitsLeft = 8;
            }
        }

        // flush
        if (bitsLeft != 8) {
            info.write_value(current);
        }
    }

    __host__ __device__
    void compute_bit_width(INPUT_T* data, uint32_t len, uint8_t* hist) {
        for (int i=0; i<HIST_LEN; ++i) hist[i] = 0;
        for (uint32_t i=0; i<len; ++i) {
            hist[get_encoded_bit_width(find_closes_num_bits(data[i]))] ++;
        }
    }

    __host__ __device__
    uint8_t compute_percentile_bit_width(uint8_t* hist, uint32_t len, uint8_t p) {
        p = min(100 - p, 100);
        int32_t plen = (p == 0 ? 0 : (len * p - 1) / 100 + 1);
        for (int8_t i=HIST_LEN-1; i>=0; --i) {
            plen -= hist[i];
            if (plen < 0) {
                return get_decoded_bit_width(static_cast<uint8_t>(i));
                // return X[static_cast<uint8_t>(i)];
            }
        }
        return 0;
    }

    __host__ __device__
    void preparePatchedBlob1(encode_info<>& info, patch_blob& pb) {
        // mask will be max value beyond which patch will be generated
        int64_t mask = static_cast<int64_t>(static_cast<uint64_t>(1) << pb.bits95p) - 1;
        // printf("<<<<<<<<<<<<<=== pb95p %u\n", pb.bits95p);i
        pb.patch_len = ceil(info.num_literals, static_cast<uint32_t>(20));

        pb.patch_width = get_closest_bit(pb.bits100p - pb.bits95p);


        if (pb.patch_width == 64) {
            pb.patch_width = 56;
            pb.bits95p = 8;
            mask = static_cast<INPUT_T>(static_cast<UINPUT_T>(1) << pb.bits95p) - 1;
        }

        uint32_t gapIdx = 0;
        uint32_t patchIdx = 0;
        size_t prev = 0;
        size_t maxGap = 0;

        INPUT_T gapList[MAX_LITERAL_SIZE], patchList[MAX_LITERAL_SIZE];

        for(size_t i = 0; i < info.num_literals; i++) {
            // if value is above mask then create the patch and record the gap

	    if (pb.reduced_literals[i] > mask) {
                size_t gap = i - prev;
                if (gap > maxGap) {
                    maxGap = gap;
                }

                // gaps are relative, so store the previous patched value index
                prev = i;
                gapList[gapIdx ++] = gap;

                // extract the most significant bits that are over mask bits
                INPUT_T patch = pb.reduced_literals[i] >> pb.bits95p;
                patchList[patchIdx ++] = patch;

                // strip off the MSB to enable safe bit packing
                // info.literals[i] &= mask;
                pb.reduced_literals[i] &= mask;

	    }
        }

        // adjust the patch length to number of entries in gap list
        pb.patch_len = gapIdx;
        

        // if the element to be patched is the first and only element then
        // max gap will be 0, but to store the gap as 0 we need atleast 1 bit
        if (maxGap == 0 && pb.patch_len != 0) {
            pb.patch_gap_width = 1;
        } else {
            pb.patch_gap_width = find_closes_num_bits(static_cast<INPUT_T>(maxGap));
        }

        if (pb.patch_gap_width > 8) {
            pb.patch_gap_width = 8;
            // for gap = 511, we need two additional entries in patch list
            if (maxGap == 511) {
                pb.patch_len += 2;
            } else {
                pb.patch_len += 1;
            }
        }

        // create gap vs patch list
        gapIdx = 0;
        patchIdx = 0;

        uint32_t gap_list_pos = 0;
        for(size_t i = 0; i < pb.patch_len; i++) {
            INPUT_T g = gapList[gapIdx++];
            INPUT_T p = patchList[patchIdx++];
            while (g > 255) {
                pb.gap_patch_list[gap_list_pos ++] = (255L << pb.patch_width);
                i++;
                g -= 255;
            }

            // store patch value in LSBs and gap in MSBs
            pb.gap_patch_list[gap_list_pos ++] = ((g << pb.patch_width) | p);
        }
    }

    __host__ __device__
    void determineEncoding(encode_info<>& info) {
        // printf("determine encoding\n");
        INPUT_T* literals = info.literals;
        if (info.num_literals <= MINIMUM_REPEAT) {
            writeDirectValues(info);
            return;
        }

        // for identifying monotonic sequences
        bool isIncreasing = true;
        bool isDecreasing = true;
        info.is_fixed_delta = true;

        INPUT_T literal_min = literals[0], literal_max = literals[0];
        INPUT_T initialDelta = literals[1] - literals[0];
        INPUT_T currDelta = 0;
        INPUT_T max_delta = 0;

        INPUT_T* deltas = info.deltas;
        for (size_t i = 1; i < info.num_literals; i++) {
            const INPUT_T l1 = literals[i];
            const INPUT_T l0 = literals[i - 1];
            currDelta = l1 - l0;
            literal_min = min(literal_min, l1);
            literal_max = max(literal_max, l1);

            isIncreasing &= (l0 <= l1);
            isDecreasing &= (l0 >= l1);

            info.is_fixed_delta &= (currDelta == initialDelta);
            deltas[i - 1] = (currDelta);
            max_delta = max(max_delta, deltas[i - 1]);
        }
        deltas[0] = literals[1] - literals[0]; // Initial

        // it's faster to exit under delta overflow condition without checking for
        // PATCHED_BASE condition as encoding using DIRECT is faster and has less
        // overhead than PATCHED_BASE
        if (!is_safe_subtract(literal_max, literal_min)) {
	    writeDirectValues(info);
            return;
        }

        if (literal_min == literal_max || info.is_fixed_delta) {
#ifdef DEBUG
if (info.cid == ERR_CHUNK && info.tid == ERR_THREAD) {
	printf("thread %u case write delta case 2\n", info.tid);
}
#endif
            writeDeltaValues(info);
            return;
        }


        if (initialDelta != 0) {
            info.delta_bits = find_closes_num_bits(max_delta);

            if (isIncreasing || isDecreasing) {
#ifdef DEBUG
if (info.cid == ERR_CHUNK && info.tid == ERR_THREAD) {
	printf("thread %u case write delta case 1\n", info.tid);
}
#endif

	writeDeltaValues(info);
                return;
            }
        }


        uint8_t hist[HIST_LEN];
        compute_bit_width(info.literals, info.num_literals, hist);
        auto bits100p = compute_percentile_bit_width(hist, info.num_literals, 100);
        auto bits90p = compute_percentile_bit_width(hist, info.num_literals, 90);

        uint32_t diffBitsLH = bits100p - bits90p;

        // if the difference between 90th percentile and 100th percentile fixed
        // bits is > 1 then we need patch the values
        if (diffBitsLH > 1) {
            patch_blob pb;
            pb.literal_min = literal_min;
	    for (size_t i = 0; i < info.num_literals; i++) {
                pb.reduced_literals[i] = literals[i] - literal_min;
            }

            compute_bit_width(pb.reduced_literals, info.num_literals, hist);
            pb.bits95p = compute_percentile_bit_width(hist, info.num_literals, 95);
            pb.bits100p = compute_percentile_bit_width(hist, info.num_literals, 100);

            // after base reducing the values, if the difference in bits between
            // 95th percentile and 100th percentile value is zero then there
            // is no point in patching the values, in which case we will
            // fallback to DIRECT encoding.
            // The decision to use patched base was based on zigzag values, but the
            // actual patching is done on base reduced literals.
	    
            if ((pb.bits100p - pb.bits95p) != 0) {
            
	    // if (false) {

		 preparePatchedBlob1(info, pb);
                writePatchedBasedValues(info, pb);
                return;
            } else {

		    writeDirectValues(info);
                return;
            }
        } else {
            // if difference in bits between 95th percentile and 100th percentile is
            // 0, then patch length will become 0. Hence we will fallback to direct

	    writeDirectValues(info);
            return;
        }
    }

    __host__ __device__
    inline void write_header_short_repeat(const uint8_t width, const uint8_t count, encode_info<>& info) {
        info.write_value(((width - 1) << 3 ) | (count - MINIMUM_REPEAT));
    }

    __host__ __device__
    void writeShortRepeatValues(encode_info<>& info) {
        // printf("write short repeat\n");
        INPUT_T	val = info.literals[0];

        const uint8_t val_bits = find_closes_num_bits(val);
        const uint8_t val_bytes = (val_bits - 1) / 8 + 1;


        write_header_short_repeat(val_bytes, info.fix_runlen, info);
        // info.write_value(((val_bytes - 1) << 3 ) | (info.fix_runlen - MINIMUM_REPEAT));

        // store in big endianess
        for(int32_t i = static_cast<int32_t>(val_bytes - 1); i >= 0; i--) {
            info.write_value(static_cast<uint8_t>((val >> (i * 8)) & 0xff));
        }

        info.fix_runlen = 0;
        info.num_literals = 0;
        info.var_runlen = 0;
    }

    __host__ __device__
    inline void write_header_direct(const uint16_t encoded_width, const uint16_t len, encode_info<>& info) {
        const uint8_t first = static_cast<uint8_t>(0b01000000 | (encoded_width << 1) | ((len & 0x100) >> 8));
        const uint8_t second = static_cast<uint8_t>(len & 0xff);
        info.write_value(first);
        info.write_value(second);
    }

    __host__ __device__
    void writeDirectValues(encode_info<>& info) {
        // printf("write direct\n");

        // write the number of fixed bits required in next 5 bits
        uint8_t hist[HIST_LEN];
        compute_bit_width(info.literals, info.num_literals, hist);
        auto num_bits = compute_percentile_bit_width(hist, info.num_literals, 100);
        num_bits = get_closest_aligned_bit(num_bits);

        const auto encoded_num_bits = get_encoded_bit_width(num_bits);
        const auto var_len = info.var_runlen - 1;

        write_header_direct(encoded_num_bits, var_len, info);
        write_aligned_ints(info.literals, info.num_literals, num_bits, info);

        // reset run length
        info.var_runlen = 0;
        info.num_literals = 0;
    }

    __host__ __device__ 
    void writeVulong(int64_t val, encode_info<>& info) {
        while (true) {
        if ((val & ~0x7f) == 0) {
            info.write_value(static_cast<char>(val));
            return;
        } else {
            info.write_value(static_cast<char>(0x80 | (val & 0x7f)));
            val = (static_cast<uint64_t>(val) >> 7);
            // val = (val >> 7);
        }
        }
    }

    __host__ __device__
    void writeVslong(int64_t val, encode_info<>& info) {
        writeVulong((val << 1) ^ (val >> 63), info);
    }

    __host__ __device__ 
    inline void write_header_delta(const uint8_t encoded_width, const uint16_t len, encode_info<>& info) {
        const uint8_t first = 0b11000000 | (encoded_width << 1) | ((len & 0x100) >> 8);
        const uint8_t second = len & 0xff;
        info.write_value(first);
        info.write_value(second);
    }

    __host__ __device__ 
    void writeDeltaValues(encode_info<>& info) {
        // printf("write delta\n");
#ifdef DEBUG
if (info.cid == ERR_CHUNK && info.tid == ERR_THREAD) {
	printf("thread %u case write delta values\n", info.tid);
    for (int i=0; i<info.num_literals-1; ++i) {
        printf("delta[%d]: %d\n", i, info.deltas[i]);
    }
}
#endif
        uint16_t len = 0;
        uint8_t encoded_width = 0;
        uint8_t num_bits = get_closest_aligned_bit(info.delta_bits);

        if (info.is_fixed_delta) {
            if (info.fix_runlen > MINIMUM_REPEAT) {
                // ex. sequence: 2 2 2 2 2 2 2 2
                len = info.fix_runlen - 1;
                info.fix_runlen = 0;
            } else {
                // ex. sequence: 4 6 8 10 12 14 16
                len = info.var_runlen - 1;
                info.var_runlen = 0;
            }
        } else {
            // fixed width 0 is used for long repeating values.
            // sequences that require only 1 bit to encode will have an additional bit
            if (num_bits == 1) {
                num_bits = 2;
            }
            encoded_width = get_encoded_bit_width(num_bits);
            len = info.var_runlen - 1;
            info.var_runlen = 0;
        }

        write_header_delta(encoded_width, len, info);
        writeVulong(info.literals[0], info);

        writeVslong(info.deltas[0], info);
#ifdef DEBUG
        if (info.cid==ERR_CHUNK && info.tid==ERR_THREAD)printf("thread %d write initial delta %d\n", info.tid, info.deltas[0]);
#endif
        if (!info.is_fixed_delta) {
            write_aligned_ints(info.deltas + 1, info.num_literals - 2, num_bits, info);
        }

        for (int i=0; i<info.num_literals; ++i) {
            info.deltas[i] = 0;
        }
        info.num_literals = 0;
        // info.fix_runlen = 0;
        info.var_runlen = 0;
    }
    
    __host__ __device__ 
    void writePatchedBasedValues(encode_info<>& info, patch_blob& pb) {
        uint32_t& varlen = info.var_runlen;
        varlen -= 1;
        const uint8_t headerFirstByte = static_cast<uint8_t>(
            0b10000000 | 
            (get_encoded_bit_width(pb.bits95p) << 1) | 
            ((varlen & 0x100) >> 8));


        // 8 bit from 9 bit length
        const uint8_t headerSecondByte = static_cast<uint8_t>(varlen & 0xff);

        // if the min value is negative toggle the sign
        const bool isNegative = (pb.literal_min < 0);
        if (isNegative) {
            pb.literal_min = -pb.literal_min;
        }

        const uint32_t val_bits = find_closes_num_bits(pb.literal_min) + 1;
        const uint32_t val_bytes = (val_bits - 1) / 8 + 1;

        // if the base value is negative then set MSB to 1
        if (isNegative) {
            pb.literal_min |= (1LL << ((val_bytes * 8) - 1));
        }

#ifdef DEBUG
if (info.cid == ERR_CHUNK && info.tid == ERR_THREAD) {
printf("header patched with pb %d with fbw %d \n", pb.patch_width, (get_encoded_bit_width(pb.bits95p)));
}
#endif

        const char headerThirdByte = static_cast<char>(((val_bytes - 1) << 5) | get_encoded_bit_width(pb.patch_width));
        const char headerFourthByte = static_cast<char>((pb.patch_gap_width - 1) << 5 | pb.patch_len);

        info.write_value(headerFirstByte);
        info.write_value(headerSecondByte);
        info.write_value(headerThirdByte);
        info.write_value(headerFourthByte);

        // write the base value using fixed bytes in big endian order
        for(int32_t i = static_cast<int32_t>(val_bytes - 1); i >= 0; i--) {
            char b = static_cast<char>(((pb.literal_min >> (i * 8)) & 0xff));
            info.write_value(b);
        }

        // base reduced literals are bit packed
        uint32_t closestFixedBits = get_closest_bit(pb.bits95p);

        // for (int i=0; i< info.num_literals; ++i) {
        //     printf("lit:%ld\n", pb.reduced_literals[i]);
        // }

        write_unaligned_ints(pb.reduced_literals, info.num_literals, closestFixedBits, info); //TODO
        closestFixedBits = get_closest_bit(pb.patch_gap_width + pb.patch_width);
        // for (int i=0; i<pb.patch_len; ++i) {
        //     printf("p:%ld\n", pb.gap_patch_list[i]);
        // }
	write_unaligned_ints(pb.gap_patch_list, pb.patch_len, closestFixedBits, info); //TODO

        // reset run length
        info.var_runlen = 0;
        info.num_literals = 0;
    }

    __global__ void kernel_encode(INPUT_T* in, const uint64_t in_n_bytes, const uint32_t n_chunks, uint8_t* out, uint64_t *offset, uint64_t CHUNK_SIZE) {
        uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < n_chunks) {
            block_encode(tid, in, in_n_bytes, out, &offset[tid + 1], CHUNK_SIZE);
        }
    }

    __host__ __device__ void block_encode(const uint64_t tid, INPUT_T* in, const uint64_t in_n_bytes, uint8_t* out, uint64_t* offset, uint64_t CHUNK_SIZE) {
        encode_info<> info;
        uint64_t OUTPUT_CHUNK_SIZE = CHUNK_SIZE * 2;
 	info.output = out + tid * OUTPUT_CHUNK_SIZE;
 
        INPUT_T prev_delta;

        auto& num_literals = info.num_literals;
        auto& fix_runlen = info.fix_runlen;
        auto& var_runlen = info.var_runlen;
        INPUT_T *literals = info.literals;

        const uint64_t start = tid * CHUNK_SIZE / sizeof(INPUT_T);
        const uint64_t end = min((tid + 1) * CHUNK_SIZE, in_n_bytes) / sizeof(INPUT_T);

        auto mychunk_size = end - start;
        // printf("%lu: (%lu, %lu)\n", tid, start, end);

        in += tid * CHUNK_SIZE / sizeof(INPUT_T);

        while (mychunk_size-- > 0) {
            auto val = *(in ++);
            // printf("%lu read %ld\n", tid, val);
            if (num_literals == 0) {
                prev_delta = 0;
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
                // initializeLiterals(val); // REMOVE COMMENT HERE
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
        *offset = info.potision;

        // for (int i=0; i<info.potision; ++i) {
        //     if (tid == 1) printf("thread 1 write byte %x\n", info.output[i]);
        // }

        // printf("thread %llu write out #bytes %u\n", tid, info.potision);
    }

    __host__ __device__ void shift_data(const uint8_t* in, uint8_t* out, const uint64_t* ptr, const uint64_t tid, uint64_t CHUNK_SIZE) {
        uint64_t OUTPUT_CHUNK_SIZE = 2 * CHUNK_SIZE;
        const auto* cur_in = in + tid * OUTPUT_CHUNK_SIZE;
        uint8_t* cur_out = out + ptr[tid];
        memcpy(cur_out, cur_in, sizeof(uint8_t) * (ptr[tid + 1] - ptr[tid]));
    }

    __global__ void kernel_shift_data(const uint8_t* in, uint8_t* out, const uint64_t* ptr, const uint64_t n_chunks, uint64_t CHUNK_SIZE) {
        uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < n_chunks) shift_data(in, out, ptr, tid, CHUNK_SIZE);
    }

    __host__ void compress_gpu(const INPUT_T* in, const uint64_t in_n_bytes, uint8_t*& out, uint64_t& out_n_bytes, uint64_t CHUNK_SIZE) {
        initialize_bit_maps();
        uint64_t OUTPUT_CHUNK_SIZE = CHUNK_SIZE * 2;
        uint32_t n_chunks = (in_n_bytes - 1) / CHUNK_SIZE + 1;
        INPUT_T* d_in;
        uint8_t *d_out, *d_shift;
        cuda_err_chk(cudaMalloc((void**)&d_in, in_n_bytes));
        cuda_err_chk(cudaMalloc((void**)&d_out, n_chunks * OUTPUT_CHUNK_SIZE));

        cuda_err_chk(cudaMemcpy(d_in, in, in_n_bytes, cudaMemcpyHostToDevice));

        uint64_t *d_ptr;
        cuda_err_chk(cudaMalloc((void**)&d_ptr, sizeof(uint64_t) * (n_chunks + 1)));

        const uint64_t grid_size = ceil<uint64_t>(n_chunks, BLK_SIZE);
        
        // printf("chunks: %u\n", n_chunks);
        // printf("output chunk size: %lu\n", OUTPUT_CHUNK_SIZE);

        kernel_encode<<<grid_size, BLK_SIZE>>>(d_in, in_n_bytes, n_chunks, d_out, d_ptr, CHUNK_SIZE);
        cuda_err_chk(cudaDeviceSynchronize());

        thrust::inclusive_scan(thrust::device, d_ptr, d_ptr + n_chunks + 1, d_ptr);

        uint64_t data_n_bytes;
        cuda_err_chk(cudaMemcpy(&data_n_bytes, d_ptr + n_chunks, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        // printf("output bytes: %lu\n", data_n_bytes);

        cudaMalloc((void**)&d_shift, data_n_bytes * sizeof(uint8_t));
        kernel_shift_data<<<grid_size, BLK_SIZE>>>(d_out, d_shift, d_ptr, n_chunks, CHUNK_SIZE);
        cuda_err_chk(cudaDeviceSynchronize());
    	

        const uint64_t ptr_n_bytes = sizeof(uint64_t) * (n_chunks + 1);
	    size_t exp_out_n_bytes = sizeof(uint32_t) + ptr_n_bytes + sizeof(uint64_t) + data_n_bytes;

        out = new uint8_t[exp_out_n_bytes];
        *(uint32_t*)out = (n_chunks + 1);
        cuda_err_chk(cudaMemcpy(out + sizeof(uint32_t), d_ptr, ptr_n_bytes, cudaMemcpyDeviceToHost));
        
        uint64_t* data_len = (uint64_t*)(out + sizeof(uint32_t) + ptr_n_bytes);
        *data_len = in_n_bytes;
        cuda_err_chk(cudaMemcpy((uint8_t*)(data_len + 1), d_shift, data_n_bytes, cudaMemcpyDeviceToHost));

        cuda_err_chk(cudaFree(d_shift));
        cuda_err_chk(cudaFree(d_ptr));
        cuda_err_chk(cudaFree(d_in));
        cuda_err_chk(cudaFree(d_out));

        out_n_bytes = exp_out_n_bytes;
    }

}

#endif
