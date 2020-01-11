#include <stdint.h>
// #define uint8_t char

constexpr   uint32_t LOOKAHEAD_SIZE_() { return 4096; }
constexpr   uint64_t CHUNK_SIZE_() { return 128; }
constexpr   uint64_t INPUT_SIZE_() { return 2048; }

#define LOOKAHEAD_SIZE			  LOOKAHEAD_SIZE_()			  
#define CHUNK_SIZE                CHUNK_SIZE_()
#define INPUT_SIZE                INPUT_SIZE_()

#define MAX_LITERAL_SIZE 128
#define MINIMUM_REPEAT 3
#define MAXIMUM_REPEAT (127 + MINIMUM_REPEAT)

#define OUTPUT_CHUNK_SIZE (CHUNK_SIZE + (MAX_LITERAL_SIZE - 1) / CHUNK_SIZE + 1) //maximum output chunk size

namespace brle {
    __host__ __device__ void decode(const uint8_t *in, uint8_t* out, const uint64_t *ptr, const uint64_t tid) {
        out += tid * CHUNK_SIZE;
        in += tid * OUTPUT_CHUNK_SIZE;
        
        uint64_t input_len = ptr[tid + 1] - ptr[tid];
        uint64_t input_pos = 0;

        while (input_pos < input_len) {
            uint64_t count;
            uint8_t ch = in[input_pos ++];
            if (ch > 127) {
                count = static_cast<uint64_t>(ch - 127);
                memcpy(out, in + input_pos, count * sizeof(uint8_t));
                input_pos += count;
            } else {
                count = static_cast<uint64_t>(ch) + MINIMUM_REPEAT;
                memset(out, in[input_pos ++], count * sizeof(uint8_t));
            }
            out += count;
        }
    }

    __global__ void kernel_decompress(const uint8_t* compressed, const uint64_t *pos, uint8_t* uncompressed) {
        uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
        decode(compressed, uncompressed, pos, tid);
        // ByteRleDecoder decoder(compressed + t * OUTPUT_CHUNK_SIZE, pos[t + 1] - pos[t]);
        // decoder.decode(uncompressed + t * CHUNK_SIZE);
    }

    __host__ __device__ int encode(uint8_t* in, uint8_t* out, const uint64_t in_n_bytes, const uint64_t tid) {
    #define WRITE_VALUES \
    if (num_literals != 0) { \
        if (repeat) { \
            out[pos++] = static_cast<uint8_t>(num_literals - MINIMUM_REPEAT); \
            out[pos++] = literals[0]; \
        } else { \
            out[pos++] = static_cast<uint8_t>(num_literals + 127); \
            memcpy(out + pos, literals, sizeof(uint8_t) * num_literals); \
            pos += num_literals; \
        } \
        repeat = false; \
        tail_run = 0; \
        num_literals = 0; \
    } 

        uint8_t lookahead[LOOKAHEAD_SIZE];
        uint8_t literals[MAX_LITERAL_SIZE];

        uint32_t pos = 0;
        uint32_t num_literals = 0;
        uint32_t tail_run = 0;
        bool repeat = false;

        uint32_t lookahead_head = 0;
        uint32_t lookahead_count = 0;
        uint64_t consumed_bytes = 0;
        uint64_t used_bytes = 0;

        uint32_t my_chunk_size = min(in_n_bytes - tid * CHUNK_SIZE, CHUNK_SIZE);

        while (used_bytes < my_chunk_size) {
            while ((lookahead_count < LOOKAHEAD_SIZE) && (consumed_bytes < my_chunk_size))  {
                lookahead[(lookahead_head + (lookahead_count++)) % LOOKAHEAD_SIZE] =
                in[consumed_bytes++];
            }
            
            uint8_t c = lookahead[lookahead_head];
            lookahead_head = (lookahead_head + 1) % LOOKAHEAD_SIZE;
            used_bytes ++;

            if (num_literals == 0) {
                literals[num_literals++] = c;
                tail_run = 1;
            } else if (repeat) {
                if (c == literals[0]) {
                    num_literals ++;
                    if (num_literals == MAXIMUM_REPEAT) {
                        WRITE_VALUES;
                    }
                } else {
                    WRITE_VALUES;
                    literals[num_literals++] = c;
                    tail_run = 1;
                }
            } else {
                if (c == literals[num_literals - 1]) 
                    tail_run ++;
                else 
                    tail_run = 1;

                if (tail_run == MINIMUM_REPEAT) {
                    if (num_literals + 1 == MINIMUM_REPEAT) {
                        num_literals += 1;
                    } else {
                        num_literals -= MINIMUM_REPEAT - 1;
                        WRITE_VALUES;
                        literals[0] = c;
                        num_literals = MINIMUM_REPEAT;
                    }
                    repeat = true;
                } else {
                    literals[num_literals++] = c;
                    if (num_literals == MAX_LITERAL_SIZE) {
                        WRITE_VALUES;
                    }
                }
            }
        }
        WRITE_VALUES;
    #undef WRITE_VALUES

        return pos;
    }

    __global__ void kernel_compress(uint8_t* in, uint8_t* out, uint64_t *offset, const uint64_t in_n_bytes) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        offset[tid + 1] = encode(in + tid * CHUNK_SIZE, out + tid * OUTPUT_CHUNK_SIZE, in_n_bytes, tid);
    }
}