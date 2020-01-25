#include <common.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

// #define uint8_t char

constexpr   uint32_t LOOKAHEAD_SIZE_() { return 4096; }
constexpr   uint64_t CHUNK_SIZE_() { return 256*1024; }
constexpr   uint16_t BLK_SIZE_() { return (1024); }

#define LOOKAHEAD_SIZE			  LOOKAHEAD_SIZE_()			  
#define CHUNK_SIZE                CHUNK_SIZE_()
#define BLK_SIZE			      BLK_SIZE_()			  

#define MAX_LITERAL_SIZE 128
#define MINIMUM_REPEAT 3
#define MAXIMUM_REPEAT (127 + MINIMUM_REPEAT)

#define OUTPUT_CHUNK_SIZE (CHUNK_SIZE + (CHUNK_SIZE - 1) / MAX_LITERAL_SIZE + 1) //maximum output chunk size

namespace brle {
    __host__ __device__ void decode(const uint8_t *in, uint8_t* out, const uint64_t *ptr, const uint64_t tid) {
        out += tid * CHUNK_SIZE;
        in += ptr[tid];
        
        uint64_t input_len = ptr[tid + 1] - ptr[tid];
        uint64_t input_pos = 0;

        while (input_pos < input_len) {
            // printf("pos:%ld len:%ld\n", input_pos, input_len);
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

    __global__ void kernel_decompress(const uint8_t* compressed, const uint64_t *pos, uint8_t* uncompressed, const uint32_t n_chunks) {
        uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < n_chunks) {
            decode(compressed, uncompressed, pos, tid);
        }
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

            bool should_break = false;
            while (lookahead_count-- > 0) {
                uint8_t c = lookahead[lookahead_head];
                used_bytes ++;

                lookahead_head = (lookahead_head + 1) % LOOKAHEAD_SIZE;
                
                if (num_literals == 0) {
                    literals[num_literals++] = c;
                    tail_run = 1;
                } else if (repeat) {
                    if (c == literals[0]) {
                        num_literals ++;
                        if (num_literals == MAXIMUM_REPEAT) {
                            WRITE_VALUES;
                            should_break = true;
                        }
                    } else {
                        WRITE_VALUES;
                        literals[num_literals++] = c;
                        tail_run = 1;
                        should_break = true;
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
                            should_break = true;
                        }
                        repeat = true;
                    } else {
                        literals[num_literals++] = c;
                        if (num_literals == MAX_LITERAL_SIZE) {
                            WRITE_VALUES;
                            should_break = true;
                        }
                    }
                }
                if (should_break) break;
            }
        }
        WRITE_VALUES;
    #undef WRITE_VALUES

        return pos;
    }

    __host__ __device__ void shift_data(const uint8_t* in, uint8_t* out, const uint64_t* ptr, const uint64_t tid) {
        const uint8_t* cur_in = in + tid * OUTPUT_CHUNK_SIZE;
        uint8_t* cur_out = out + ptr[tid];
        memcpy(cur_out, cur_in, sizeof(uint8_t) * ptr[tid + 1] - ptr[tid]);
    }

    __global__ void kernel_shift_data(const uint8_t* in, uint8_t* out, const uint64_t* ptr, const uint64_t n_chunks) {
        uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < n_chunks)
            shift_data(in, out, ptr, tid);
    }

    __global__ void kernel_compress(uint8_t* in, uint8_t* out, uint64_t *offset, const uint64_t in_n_bytes, const uint32_t n_chunks) {
        uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < n_chunks) {
            offset[tid + 1] = encode(in + tid * CHUNK_SIZE, out + tid * OUTPUT_CHUNK_SIZE, in_n_bytes, tid);
        }
    }

    __host__ void compress_gpu(const uint8_t* in, uint8_t** out, const uint64_t in_n_bytes, uint64_t *out_n_bytes) {
        uint32_t n_chunks = (in_n_bytes - 1) / CHUNK_SIZE + 1;
        
        uint8_t* d_in, *d_out, *d_shift;
        cuda_err_chk(cudaMalloc((void**)&d_in, in_n_bytes));
        cuda_err_chk(cudaMalloc((void**)&d_out, n_chunks * OUTPUT_CHUNK_SIZE));

        cudaMemcpy(d_in, in, in_n_bytes, cudaMemcpyHostToDevice);

        uint64_t *d_ptr;
        cuda_err_chk(cudaMalloc((void**)&d_ptr, sizeof(uint64_t) * (n_chunks + 1)));


        uint64_t grid_size = ceil<uint64_t>(n_chunks, BLK_SIZE);
        
        kernel_compress<<<grid_size, BLK_SIZE>>>(d_in, d_out, d_ptr, in_n_bytes, n_chunks);
        thrust::inclusive_scan(thrust::device, d_ptr, d_ptr + n_chunks + 1, d_ptr);

        uint64_t data_n_bytes;
        cudaMemcpy(&data_n_bytes, d_ptr + n_chunks, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cuda_err_chk(cudaMalloc((void**)&d_shift, data_n_bytes * sizeof(uint8_t)));
        kernel_shift_data<<<grid_size, BLK_SIZE>>>(d_out, d_shift, d_ptr, n_chunks);

        cuda_err_chk(cudaFree(d_in));
        cuda_err_chk(cudaFree(d_out));
    	cuda_err_chk(cudaDeviceSynchronize());

        size_t len_n_bytes = sizeof(uint32_t); // length of chunk
        size_t ptr_n_bytes = sizeof(uint64_t) * (n_chunks + 1); // device pointer 
	    size_t exp_out_n_bytes = len_n_bytes + ptr_n_bytes + sizeof(uint64_t) + data_n_bytes;

        *out = new uint8_t[exp_out_n_bytes];
        uint32_t* ptr_len = (uint32_t*)*out;
        *ptr_len = (n_chunks + 1);

        cuda_err_chk(cudaMemcpy(*out + len_n_bytes, d_ptr, ptr_n_bytes, cudaMemcpyDeviceToHost));

        
        uint64_t* out_len = (uint64_t*)(*out + len_n_bytes + ptr_n_bytes);
        *out_len = in_n_bytes;

        cuda_err_chk(cudaMemcpy(*out + len_n_bytes + ptr_n_bytes + sizeof(uint64_t), d_shift, data_n_bytes, cudaMemcpyDeviceToHost));
        cuda_err_chk(cudaFree(d_shift));

        // uint8_t* ptr = (uint8_t*)(*out + len_n_bytes + ptr_n_bytes + sizeof(uint64_t));
        // for (int i=0; i<data_n_bytes; ++i) {
        //     printf("output:%d\n", ptr[i]);
        // }
        // printf("expect:%lu\n", exp_out_n_bytes);
        // printf("n_chunks:%u\n", n_chunks);

        *out_n_bytes = exp_out_n_bytes;
    }

    __host__ void decompress_gpu(const uint8_t* const in, uint8_t** out, const uint64_t in_n_bytes, uint64_t* out_n_bytes) {
        uint8_t* d_in, *d_out;
        uint64_t* d_ptr;
        uint32_t n_ptr = *((uint32_t*)in);
        uint32_t n_chunks = n_ptr - 1;

        uint64_t header_bytes = sizeof(uint32_t) + sizeof(uint64_t) * n_ptr;
        uint64_t data_n_bytes = in_n_bytes - header_bytes;

        uint64_t out_bytes;

        const uint64_t* ptr = (const uint64_t*)(in + sizeof(uint32_t));
        cuda_err_chk(cudaMalloc((void**)&d_in, data_n_bytes));

        out_bytes = *(uint64_t*)(in + header_bytes);

        cuda_err_chk(cudaMemcpy(d_in, in + header_bytes + sizeof(uint64_t), data_n_bytes, cudaMemcpyHostToDevice));
        cuda_err_chk(cudaMalloc((void**)&d_out, CHUNK_SIZE * n_chunks));
        cuda_err_chk(cudaMalloc((void**)&d_ptr, sizeof(uint64_t) * n_ptr));
        cuda_err_chk(cudaMemcpy(d_ptr, ptr, sizeof(uint64_t) * n_ptr, cudaMemcpyHostToDevice));

        uint64_t grid_size = ceil<uint64_t>(n_chunks, BLK_SIZE);
        kernel_decompress<<<grid_size, BLK_SIZE>>>(d_in, d_ptr, d_out, n_chunks);

	    cuda_err_chk(cudaDeviceSynchronize());

        // printf("actual:%lu\n", out_bytes);

        *out = new uint8_t[out_bytes];
	    cuda_err_chk(cudaMemcpy(*out, d_out, out_bytes, cudaMemcpyDeviceToHost));


	    cuda_err_chk(cudaFree(d_in));
	    cuda_err_chk(cudaFree(d_out));
	    cuda_err_chk(cudaFree(d_ptr));
        
        *out_n_bytes = out_bytes;
    }
}
