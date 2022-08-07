
#include <common.h>
#include <fstream>
#include <sys/types.h> 
#include <sys/stat.h> 
#include <fcntl.h>
#include <sys/mman.h>
#include <simt/atomic>
#include <iostream>

#include <common_warp.h>


//#include <cuda/atomic>
#include "utils.h"

#define FULL_MASK 0xFFFFFFFF



// template <typename T>
// struct queue {
//     T* queue_;
//     simt::atomic<uint8_t, simt::thread_scope_block>* head;
//     simt::atomic<uint8_t, simt::thread_scope_block>* tail;
//     uint8_t len;

//     __device__
//     queue(T* buff, simt::atomic<uint8_t, simt::thread_scope_block>* h, simt::atomic<uint8_t, simt::thread_scope_block>* t, const uint8_t l) {
//         queue_ = buff;
//         head = h;
//         tail = t;
//         len = l;

//     }

//     __device__
//     void enqueue(const T* v) {

//         const auto cur_tail = tail->load(simt::memory_order_relaxed);
//         const auto next_tail = (cur_tail + 1) % len;

//         while (next_tail == head->load(simt::memory_order_acquire))
//             __nanosleep(1000);


//         queue_[cur_tail] = *v;
//         tail->store(next_tail, simt::memory_order_release);

//     }

//     __device__
//     void dequeue(T* v) {
//         //if(threadIdx.x == 0 && blockIdx.x == 0)      printf("deque start\n");

//         const auto cur_head = head->load(simt::memory_order_relaxed);
//         while (cur_head == tail->load(simt::memory_order_acquire))
//             __nanosleep(1000);

//         *v = queue_[cur_head];

//         const auto next_head = (cur_head + 1) % len;

//         head->store(next_head, simt::memory_order_release);

//     }


    
// };


template<typename T>
struct  decomp_write_queue_ele{
    T data;
    int header;
    bool done;
};



// template <typename COMP_COL_TYPE >
// struct decompress_input {


//     uint16_t row_offset;
//     uint16_t len;
//     uint16_t read_bytes;
//     COMP_COL_TYPE* pointer;



//     __device__
//     decompress_input(const uint8_t* ptr, const uint16_t l) :
//         pointer((COMP_COL_TYPE*) ptr), len(l) {
//         row_offset = 0;
//         read_bytes = 0;
//     }



//     __forceinline__
//     __device__
//     int8_t comp_read_data(const uint32_t alivemask, COMP_COL_TYPE* v) {

//         int8_t read_count  = 0;
//         bool read = (read_bytes) < len;
//         uint32_t read_sync = __ballot_sync(alivemask, read);
      
//         if (__builtin_expect (read_sync == 0, 0)){
//            // read_count = -1;
//             return -1;
//         }
        
//         //if (__builtin_expect (read, 1)) {
//         if(read){
//             *v = pointer[row_offset + __popc(read_sync & (0xffffffff >> (32 - threadIdx.x)))];
//             row_offset += __popc(read_sync);
//            // if(threadIdx.x == 0 && blockIdx.x == 0) printf("row offset: %llu\n", (unsigned long long) row_offset);
//             read_bytes += sizeof(COMP_COL_TYPE);
//             read_count = sizeof(COMP_COL_TYPE);
//         }

//         __syncwarp(alivemask);

//         return read_count;
//     }



// };



template <typename READ_TYPE>
struct input_stream_1warp
{   
    uint64_t col_len_track[32];

    uint8_t* read_ptr;
    int COMP_WRITE_BYTES;
    uint64_t read_offset;
    uint64_t out_bytes;

    __device__
    input_stream_1warp(const uint64_t* col_len, int WRITE_BYTES, uint8_t* ptr){
        read_ptr = ptr;
        COMP_WRITE_BYTES = WRITE_BYTES;
        read_offset = threadIdx.x * sizeof(READ_TYPE);
        out_bytes = 0;
        for(int i = 0; i < 32; i++){
          
            col_len_track[i] = ((col_len[ i] + COMP_WRITE_BYTES - 1) / COMP_WRITE_BYTES) * COMP_WRITE_BYTES ;
        }
        for(int i = 0; i < threadIdx.x; i++){
            col_len_track[i] -= COMP_WRITE_BYTES;
        }
    }

    __device__ 
    int8_t get_next_byte(){
        uint8_t read_data = read_ptr[read_offset];
        //read_offset += 1;
        out_bytes++;
        if(out_bytes % COMP_WRITE_BYTES != 0){
            read_offset += 1;
        }
        else{
            for(int i = 0; i < 32; i++){
                if(col_len_track[i] > 0 && i != threadIdx.x){
                    read_offset += COMP_WRITE_BYTES;
                    col_len_track[i] -= COMP_WRITE_BYTES;
                }
            }
            read_offset += 1;
        }

        return (int8_t) read_data;
    }

    
};

// //consumer of input queue
// template <typename READ_COL_TYPE, uint8_t buff_len = 4>
// struct input_stream {

//     union buff{
//         READ_COL_TYPE* b;
//         uint32_t* u;
//     }b;
//     uint8_t head;
//     uint8_t count;
//     uint8_t uint_head;
//     uint8_t uint_count;
    
    
//     queue<READ_COL_TYPE>* q;
//     uint32_t read_bytes;
//     uint32_t expected_bytes;
//     uint8_t bu_size = (sizeof(READ_COL_TYPE)* buff_len)/sizeof(uint32_t);

//     uint8_t uint_bit_offset;

//     __device__
//     input_stream(queue<READ_COL_TYPE>* q_, uint32_t eb, READ_COL_TYPE* shared_b) {
//         q = q_;
//         expected_bytes = eb;

//         b.b = shared_b;

//         head = 0;

//         uint_bit_offset = 0;
//         uint_count = 0;
//         uint_head = 0;
//         read_bytes = 0;
//         count = 0;
//         for (; (count < buff_len) && (read_bytes < expected_bytes);
//              count++, read_bytes += sizeof(READ_COL_TYPE), uint_count += sizeof(READ_COL_TYPE)/sizeof(uint32_t)) {
//              q->dequeue(b.b + count);
//             //g_dequeue<READ_COL_TYPE>(b.b + count, q);

//         }
//     }





//     template<typename T>
//     __device__
//     void get_n_bits(const uint32_t n, T* out) {

//         *out = (T) 0;

//         ((uint32_t*) out)[0] = __funnelshift_rc(b.u[(uint_head)], b.u[(uint_head+1)%bu_size], uint_bit_offset);
//         //uint32_t temp_out = __funnelshift_rc(b.u[(uint_head)], b.u[(uint_head+1)%bu_size], uint_bit_offset);


//         if (32 > n) {
//             ((uint32_t*) out)[0] <<= (32 - n);
//             ((uint32_t*) out)[0] >>= (32 - n);
//             // temp_out <<= (32 - n);
//             // temp_out >>= (32 - n);
//         }



//         uint_bit_offset += n;
//         if (uint_bit_offset >= 32) {
//             uint_bit_offset = uint_bit_offset % 32;
//             uint_head = (uint_head+1) % bu_size;
//             if ((uint_head % (sizeof(READ_COL_TYPE)/sizeof(uint32_t))) == 0) {
//                 head = (head + 1) % buff_len;
//                 count--;
//             }

//             uint_count--;

//         }
  

//     }

//     template<typename T>
//     __device__
//     void fetch_n_bits(const uint32_t n, T* out) {

//         while ((count < buff_len) && (read_bytes < expected_bytes)) {
//             //q->dequeue(b.b + ((head+count) % buff_len));
//             //g_dequeue<READ_COL_TYPE>(b.b + ((head+count) % buff_len), q);
//             q -> dequeue(b.b+((head+count) % buff_len) );
//             //q->dequeue(s_b[((head+count) % buff_len)]);
           
//             count++;
//             uint_count += sizeof(READ_COL_TYPE)/sizeof(uint32_t);
//             read_bytes += sizeof(READ_COL_TYPE);
//         }

//         get_n_bits<T>(n, out);

//     }



//     template<typename T>
//     __device__
//     void peek_n_bits(const uint32_t n, T* out) {
//         while ((count < buff_len) && (read_bytes < expected_bytes)) { 
//             q->dequeue(b.b + ((head+count) % buff_len));
//             //g_dequeue<READ_COL_TYPE>(b.b + ((head+count) % buff_len), q);

//             count++;
//             uint_count += sizeof(READ_COL_TYPE)/sizeof(uint32_t);
//             read_bytes += sizeof(READ_COL_TYPE);
//         }
//         *out = (T) 0;

//          ((uint32_t*) out)[0] = __funnelshift_rc(b.u[(uint_head)], b.u[(uint_head+1)%bu_size], uint_bit_offset);
//         //uint32_t temp_out = __funnelshift_rc(b.u[(uint_head)], b.u[(uint_head+1)%bu_size], uint_bit_offset);


//         if (32 > n) {
//             ((uint32_t*) out)[0] <<= (32 - n);
//             ((uint32_t*) out)[0] >>= (32 - n);
//             // temp_out <<= (32 - n);
//             // temp_out >>= (32 - n);
//         }



//     }


// };


template <typename DATA_TYPE>
struct decompress_output {

    DATA_TYPE* out_ptr;
    uint64_t offset;

    __device__
    decompress_output(uint8_t* ptr):
        out_ptr((DATA_TYPE*)ptr) {
        offset = 0;
    }
};


template < typename COMP_COL_TYPE>
//__forceinline__ 
__device__
void reader_warp(decompress_input<COMP_COL_TYPE>& in, queue<COMP_COL_TYPE>& rq) {
    while (true) {
        COMP_COL_TYPE v;
        int8_t rc = in.comp_read_data(FULL_MASK, &v);
        //int8_t rc = comp_read_data2(FULL_MASK, &v, in);

        if (rc == -1)
            break;
        else if (rc > 0){

            rq.enqueue(&(v));
        }
    }
}

template <typename COMP_COL_TYPE, uint8_t NUM_SUBCHUNKS>
//__forceinline__ 
__device__
void reader_warp_orig(decompress_input< COMP_COL_TYPE>& in, queue<COMP_COL_TYPE>& rq, uint8_t active_chunks) {
    //iterate number of chunks for the single reader warp
   int t = 0;
   while(true){
        bool done = true;
        for(uint8_t cur_chunk = 0; cur_chunk < active_chunks; cur_chunk++){
            COMP_COL_TYPE v;
            uint8_t rc = comp_read_data_seq<COMP_COL_TYPE>(FULL_MASK, &v, in, cur_chunk);
            if(rc != 0)
                done = false;
            
            rq.warp_enqueue(&v, cur_chunk, rc);

        }
        __syncwarp();
        if(done)
            break;
    }
}


template <typename COMP_COL_TYPE, typename DATA_TYPE, typename UDATA_TYPE, size_t in_buff_len = 32>
//__forceinline__ 
__device__
//__global__ void decompress_func_read_sync(const uint8_t* __restrict__ in, const uint64_t n_chunks, const blk_off_t* __restrict__ blk_off, const col_len_t* __restrict__ col_len, INPUT_T* __restrict__ out, uint64_t CHUNK_SIZE) {

//void decoder_warp(input_stream<COMP_COL_TYPE, in_buff_len>& s,  decompress_output<DATA_TYPE>& out, uint64_t CHUNK_SIZE, DATA_TYPE* out_buf, int COMP_COL_LEN, int READ_UNIT) {
void decoder_warp(input_stream<COMP_COL_TYPE, in_buff_len>& s, DATA_TYPE* out,   DATA_TYPE* out_buf, int COMP_COL_LEN, int READ_UNIT, const uint64_t* const col_len, uint64_t CHUNK_SIZE) {

            __shared__ DATA_TYPE out_buffer[WARP_SIZE][WRITE_VEC_SIZE];

            int tid = threadIdx.x;
            int cid = blockIdx.x;
            DATA_TYPE* out_8B = out + (blockIdx.x * CHUNK_SIZE / sizeof(DATA_TYPE) + tid * READ_UNIT);
            DATA_TYPE* base_out = out + blockIdx.x * CHUNK_SIZE / sizeof(DATA_TYPE);
            uint32_t out_ptr = 0;

            uint16_t out_buffer_ptr = 0;
            uint16_t out_counter = 0;
            uint32_t used_bytes = 0;

            auto deque_int = [&]() {
                *reinterpret_cast<VEC_T*>(out_8B + out_counter) = *reinterpret_cast<VEC_T*>(out_buffer[tid]);
                
                out_counter += WRITE_VEC_SIZE;
                if (out_counter == READ_UNIT) {
                    out_counter = 0;
                    out_8B += BLK_SIZE * READ_UNIT;
                }    
                out_buffer_ptr = 0;
            };

            auto write_int = [&](DATA_TYPE i) {
                out_ptr ++;

                if (false) {
                    if (out_buffer_ptr == WRITE_VEC_SIZE) {
                        deque_int();
                    }
                    out_buffer[tid][out_buffer_ptr++] = i;
                } else {

                    *(out_8B + out_buffer_ptr) = i; 
                    out_buffer_ptr ++;
                    if (out_buffer_ptr == READ_UNIT) {
                        out_buffer_ptr = 0;
                        out_8B += BLK_SIZE * READ_UNIT;
                    }
                }
            };
            
            auto read_byte = [&]() {
                int32_t temp_b = 0;
                s.template fetch_n_bits<int32_t>(8, &temp_b);
                uint8_t ret = (uint8_t)(temp_b & 0x00FF);
                //if(threadIdx.x == 0 && blockIdx.x == 0) printf("read: %u used_bytes: %llu \n", ret, used_bytes);
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

                return ret >> 1 ^ -(ret & 1);
            };
            
            uint32_t mychunk_size = col_len[cid * BLK_SIZE + tid];

            while (used_bytes < mychunk_size) {
                auto first = read_byte();

                switch(first & HEADER_MASK) {
                case HEADER_SHORT_REPEAT: {
                    auto nbytes = ((first >> 3) & 0x07) + 1;
                    auto count = (first & 0x07) + MINIMUM_REPEAT;
                    DATA_TYPE tmp_int = 0;
                    while (nbytes-- > 0) {
                        tmp_int |= ((DATA_TYPE)read_byte() << (nbytes * 8));
                    }
                    while (count-- > 0) {
                        write_int(tmp_int);
                    }
                } break;
                case HEADER_DIRECT: {

                    uint8_t encoded_fbw = (first >> 1) & 0x1f;
                    uint8_t fbw = get_decoded_bit_width(encoded_fbw);
                    uint8_t len = ((static_cast<uint16_t>(first & 0x01) << 8) | read_byte()) + 1;
                    uint8_t bits_left = 0 /* bits left over from unused bits of last byte */, curr_byte = 0;
                    while (len-- > 0) {
                        UDATA_TYPE result = 0;
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

                        write_int(static_cast<DATA_TYPE>(result));
                    }
                } break;
                case HEADER_DELTA: {

                    uint8_t encoded_fbw = (first >> 1) & 0x1f;
                    uint8_t fbw = get_decoded_bit_width(encoded_fbw);
                    uint8_t len = ((static_cast<uint16_t>(first & 0x01) << 8) | read_byte()) + 1;

                    DATA_TYPE base_val = static_cast<DATA_TYPE>(read_uvarint());
                    DATA_TYPE base_delta = static_cast<DATA_TYPE>(read_svarint());
                    write_int(base_val);
                    base_val += base_delta;
                    write_int(base_val);

                    len -= 2;
                    int multiplier = (base_delta > 0 ? 1 : -1);
                    if (encoded_fbw != 0) {
                        uint8_t bits_left = 0, curr_byte = 0;
                        while (len-- > 0) {
                            UDATA_TYPE result = 0;
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

                            DATA_TYPE dlt = static_cast<DATA_TYPE>(result) * multiplier;
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
                        base_val |= ((DATA_TYPE)read_byte() << (bw * 8));
                    }

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
                        
                        write_int(static_cast<DATA_TYPE>(result) + base_val);
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

                        uint32_t direct_out_ptr = base_out_ptr + patch_gap;
                        DATA_TYPE *pb_ptr = nullptr;
                        if (true) {
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
                        *pb_ptr |= (static_cast<DATA_TYPE>(result & patch_mask) << fbw);
                        *pb_ptr += base_val;
                    }
                } break;
                }
            }

}

template <typename COMP_COL_TYPE, typename DATA_TYPE, typename UDATA_TYPE, int NUM_CHUNKS, size_t in_buff_len = 32>
//__forceinline__ 
__device__
void decoder_warp_orig_dw(input_stream<COMP_COL_TYPE, in_buff_len>& s, DATA_TYPE* out,  int COMP_COL_LEN, int READ_UNIT, const uint64_t* const col_len, uint64_t CHUNK_SIZE, uint64_t chunk_id) {

            __shared__ DATA_TYPE out_buffer[WARP_SIZE][WRITE_VEC_SIZE];

            int tid = threadIdx.x;
            int cid = blockIdx.x;


            //DATA_TYPE* out_8B = out + ((uint64_t)blockIdx.x * (CHUNK_SIZE / sizeof(DATA_TYPE) )) ;
            // DATA_TYPE* base_out = out + ((uint64_t)blockIdx.x * (CHUNK_SIZE / sizeof(DATA_TYPE) ));

            DATA_TYPE* out_8B = out;
            DATA_TYPE* base_out = out;

            uint64_t out_ptr = 0;

            uint64_t out_buffer_ptr = 0;
            uint64_t out_counter = 0;
            uint64_t used_bytes = 0;

            auto deque_int = [&]() {
                *reinterpret_cast<VEC_T*>(out_8B + out_counter) = *reinterpret_cast<VEC_T*>(out_buffer[tid]);
                
                out_counter += WRITE_VEC_SIZE;
                // if (out_counter == READ_UNIT) {
                //     out_counter = 0;
                //     out_8B += BLK_SIZE * READ_UNIT;
                // }    
                out_buffer_ptr = 0;
            };

            auto write_int = [&](DATA_TYPE i) {
                out_ptr ++;

                if (false) {
                    if (out_buffer_ptr == WRITE_VEC_SIZE) {
                        deque_int();
                    }
                    out_buffer[tid][out_buffer_ptr++] = i;
                } else {


                    if(threadIdx.x == 0) {*(out_8B + out_buffer_ptr) = i; 
                     //if(blockIdx.x == 112189)   printf("data: %llx out: %p out_8B: %p bp: %lu\n", (unsigned long long) i, out, out_8B, (unsigned long)out_buffer_ptr);
                    }
                    
                    out_buffer_ptr ++;
                    // if (out_buffer_ptr == READ_UNIT) {
                    //     out_buffer_ptr = 0;
                    //     out_8B += BLK_SIZE * READ_UNIT;
                    // }
                }
            };
            
            auto read_byte = [&]() {
                int32_t temp_b = 0;
                if(threadIdx.x == 0) s.template fetch_n_bits<int32_t>(8, &temp_b);
                temp_b =  __shfl_sync(FULL_MASK, temp_b, 0);

                //if(threadIdx.x == 0 && blockIdx.x ==0) printf("temp_b: %u\n", temp_b);

                uint8_t ret = (uint8_t)(temp_b & 0x00FF);
                //if(threadIdx.x == 0 && blockIdx.x == 0) printf("read: %u used_bytes: %llu \n", ret, used_bytes);
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

                return ret >> 1 ^ -(ret & 1);
            };
            
            uint64_t mychunk_size = col_len[chunk_id];

            while (used_bytes < mychunk_size) {
                auto first = read_byte();

                switch(first & HEADER_MASK) {
                case HEADER_SHORT_REPEAT: {
                    auto nbytes = ((first >> 3) & 0x07) + 1;
                    auto count = (first & 0x07) + MINIMUM_REPEAT;
                    DATA_TYPE tmp_int = 0;
                    while (nbytes-- > 0) {
                        tmp_int |= ((DATA_TYPE)read_byte() << (nbytes * 8));
                    }
                    while (count-- > 0) {
                        write_int(tmp_int);
                    }
                } break;
                case HEADER_DIRECT: {

                    uint8_t encoded_fbw = (first >> 1) & 0x1f;
                    uint8_t fbw = get_decoded_bit_width(encoded_fbw);
                    uint8_t len = ((static_cast<uint16_t>(first & 0x01) << 8) | read_byte()) + 1;
                    uint8_t bits_left = 0 /* bits left over from unused bits of last byte */, curr_byte = 0;
                    while (len-- > 0) {
                        UDATA_TYPE result = 0;
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

                        write_int(static_cast<DATA_TYPE>(result));
                    }
                } break;
                case HEADER_DELTA: {

                    uint8_t encoded_fbw = (first >> 1) & 0x1f;
                    uint8_t fbw = get_decoded_bit_width(encoded_fbw);
                    uint8_t len = ((static_cast<uint16_t>(first & 0x01) << 8) | read_byte()) + 1;

                    DATA_TYPE base_val = static_cast<DATA_TYPE>(read_uvarint());
                    DATA_TYPE base_delta = static_cast<DATA_TYPE>(read_svarint());
                    write_int(base_val);
                    base_val += base_delta;
                    write_int(base_val);

                    len -= 2;
                    int multiplier = (base_delta > 0 ? 1 : -1);
                    if (encoded_fbw != 0) {
                        uint8_t bits_left = 0, curr_byte = 0;
                        while (len-- > 0) {
                            UDATA_TYPE result = 0;
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

                            DATA_TYPE dlt = static_cast<DATA_TYPE>(result) * multiplier;
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
                        base_val |= ((DATA_TYPE)read_byte() << (bw * 8));
                    }

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
                        
                        write_int(static_cast<DATA_TYPE>(result) + base_val);
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

                        uint32_t direct_out_ptr = base_out_ptr + patch_gap;
                        DATA_TYPE *pb_ptr = nullptr;
                        if (true) {
                            //pb_ptr = &base_out[(direct_out_ptr / READ_UNIT) * BLK_SIZE * READ_UNIT + (direct_out_ptr % READ_UNIT) + tid * READ_UNIT];

                            pb_ptr = &base_out[(direct_out_ptr)];

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
                        if(threadIdx.x == 0 ){
                            *pb_ptr -= base_val;
                            *pb_ptr |= (static_cast<DATA_TYPE>(result & patch_mask) << fbw);
                            *pb_ptr += base_val;
                        }
                    }
                } break;
                }
            }

}


template <typename COMP_COL_TYPE, typename DATA_TYPE, typename UDATA_TYPE, int NUM_CHUNKS, size_t in_buff_len = 32>
//__forceinline__ 
__device__
void decoder_warp_orig_rdw(full_warp_input_stream<COMP_COL_TYPE, in_buff_len>& s, DATA_TYPE* out,  int COMP_COL_LEN, int READ_UNIT, const uint64_t* const col_len, uint64_t CHUNK_SIZE, uint64_t chunk_id) {

            __shared__ DATA_TYPE out_buffer[WARP_SIZE][WRITE_VEC_SIZE];

            int tid = threadIdx.x;
            int cid = blockIdx.x;


            //DATA_TYPE* out_8B = out + ((uint64_t)blockIdx.x * (CHUNK_SIZE / sizeof(DATA_TYPE) )) ;
            // DATA_TYPE* base_out = out + ((uint64_t)blockIdx.x * (CHUNK_SIZE / sizeof(DATA_TYPE) ));

            DATA_TYPE* out_8B = out;
            DATA_TYPE* base_out = out;

            uint64_t out_ptr = 0;

            uint64_t out_buffer_ptr = 0;
            uint64_t out_counter = 0;
            uint64_t used_bytes = 0;

            auto deque_int = [&]() {
                *reinterpret_cast<VEC_T*>(out_8B + out_counter) = *reinterpret_cast<VEC_T*>(out_buffer[tid]);
                
                out_counter += WRITE_VEC_SIZE;
                // if (out_counter == READ_UNIT) {
                //     out_counter = 0;
                //     out_8B += BLK_SIZE * READ_UNIT;
                // }    
                out_buffer_ptr = 0;
            };

            auto write_int = [&](DATA_TYPE i) {
                out_ptr ++;

                if (false) {
                    if (out_buffer_ptr == WRITE_VEC_SIZE) {
                        deque_int();
                    }
                    out_buffer[tid][out_buffer_ptr++] = i;
                } else {


                    if(threadIdx.x == 0) {*(out_8B + out_buffer_ptr) = i; 
                     //if(blockIdx.x == 0)   printf("data: %llx out: %p out_8B: %p bp: %lu\n", (unsigned long long) i, out, out_8B, (unsigned long)out_buffer_ptr);
                    }
                    
                    out_buffer_ptr ++;
                    // if (out_buffer_ptr == READ_UNIT) {
                    //     out_buffer_ptr = 0;
                    //     out_8B += BLK_SIZE * READ_UNIT;
                    // }
                }
            };
            
            auto read_byte = [&]() {
                int32_t temp_b = 0;
                 s.template fetch_n_bits<int32_t>(8, &temp_b);
                temp_b =  __shfl_sync(FULL_MASK, temp_b, 0);
               // if(threadIdx.x == 0 && blockIdx.x ==0) printf("temp_b: %u\n", temp_b);

                uint8_t ret = (uint8_t)(temp_b & 0x00FF);
                //if(threadIdx.x == 0 && blockIdx.x == 0) printf("read: %u used_bytes: %llu \n", ret, used_bytes);
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

                return ret >> 1 ^ -(ret & 1);
            };
            
            uint64_t mychunk_size = col_len[chunk_id];

            while (used_bytes < mychunk_size) {
                auto first = read_byte();

                switch(first & HEADER_MASK) {
                case HEADER_SHORT_REPEAT: {
                    auto nbytes = ((first >> 3) & 0x07) + 1;
                    auto count = (first & 0x07) + MINIMUM_REPEAT;
                    DATA_TYPE tmp_int = 0;
                    while (nbytes-- > 0) {
                        tmp_int |= ((DATA_TYPE)read_byte() << (nbytes * 8));
                    }
                    while (count-- > 0) {
                        write_int(tmp_int);
                    }
                } break;
                case HEADER_DIRECT: {

                    uint8_t encoded_fbw = (first >> 1) & 0x1f;
                    uint8_t fbw = get_decoded_bit_width(encoded_fbw);
                    uint8_t len = ((static_cast<uint16_t>(first & 0x01) << 8) | read_byte()) + 1;
                    uint8_t bits_left = 0 /* bits left over from unused bits of last byte */, curr_byte = 0;
                    while (len-- > 0) {
                        UDATA_TYPE result = 0;
                        uint8_t bits_to_read = fbw;
                        while (bits_to_read > bits_left) {
                            if(threadIdx.x == 0){
                            result <<= bits_left;
                            result |= curr_byte & ((1 << bits_left) - 1);
                            }
                            bits_to_read -= bits_left;
                            curr_byte = read_byte();
                            bits_left = 8;
                        }
                        if(threadIdx.x == 0){
                            if (bits_to_read > 0) {
                                result <<= bits_to_read;
                                bits_left -= bits_to_read;
                                result |= (curr_byte >> bits_left) & ((1 << bits_to_read) - 1);
                            }
                        }
                        result = __shfl_sync(FULL_MASK, result, 0);
                        write_int(static_cast<DATA_TYPE>(result));
                    }
                } break;
                case HEADER_DELTA: {

                    uint8_t encoded_fbw = (first >> 1) & 0x1f;
                    uint8_t fbw = get_decoded_bit_width(encoded_fbw);
                    uint8_t len = ((static_cast<uint16_t>(first & 0x01) << 8) | read_byte()) + 1;

                    DATA_TYPE base_val = static_cast<DATA_TYPE>(read_uvarint());
                    DATA_TYPE base_delta = static_cast<DATA_TYPE>(read_svarint());
                    write_int(base_val);
                    base_val += base_delta;
                    write_int(base_val);

                    len -= 2;
                    int multiplier = (base_delta > 0 ? 1 : -1);
                    if (encoded_fbw != 0) {
                        uint8_t bits_left = 0, curr_byte = 0;
                        while (len-- > 0) {
                            UDATA_TYPE result = 0;
                            uint8_t bits_to_read = fbw;
                            while (bits_to_read > bits_left) {
                                if(threadIdx.x == 0){
                                result <<= bits_left;
                                result |= curr_byte & ((1 << bits_left) - 1);
                                }
                                bits_to_read -= bits_left;
                                curr_byte = read_byte();
                                bits_left = 8;
                            }

                            if(threadIdx.x == 0){
                            if (bits_to_read > 0) {
                                result <<= bits_to_read;
                                bits_left -= bits_to_read;
                                result |= (curr_byte >> bits_left) & ((1 << bits_to_read) - 1);
                            }

                            DATA_TYPE dlt = static_cast<DATA_TYPE>(result) * multiplier;
                            base_val += dlt; 
                            write_int(base_val);
                            }
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
                        base_val |= ((DATA_TYPE)read_byte() << (bw * 8));
                    }

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
                        
                        write_int(static_cast<DATA_TYPE>(result) + base_val);
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

                        uint32_t direct_out_ptr = base_out_ptr + patch_gap;
                        DATA_TYPE *pb_ptr = nullptr;
                        if (true) {
                            //pb_ptr = &base_out[(direct_out_ptr / READ_UNIT) * BLK_SIZE * READ_UNIT + (direct_out_ptr % READ_UNIT) + tid * READ_UNIT];

                            pb_ptr = &base_out[(direct_out_ptr)];

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
                        if(threadIdx.x == 0 ){
                            *pb_ptr -= base_val;
                            *pb_ptr |= (static_cast<DATA_TYPE>(result & patch_mask) << fbw);
                            *pb_ptr += base_val;
                        }
                    }
                } break;
                }
            }

}

template <typename COMP_COL_TYPE, typename DATA_TYPE, typename UDATA_TYPE, typename OUT_COL_TYPE, uint16_t queue_size = 4>
__global__ void 
//__launch_bounds__ (96, 13)
//__launch_bounds__ (128, 11)
inflate(uint8_t* comp_ptr, uint8_t* out, const uint64_t* const col_len_ptr, const uint64_t* const blk_offset_ptr,  int COMP_COL_LEN, int READ_UNIT, uint64_t CHUNK_SIZE) {

    __shared__ COMP_COL_TYPE in_queue_[32][queue_size];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> h[32];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> t[32];


    __shared__ COMP_COL_TYPE local_queue[32][2];

    uint64_t col_len = (col_len_ptr[32 * (blockIdx.x) + threadIdx.x]);

    h[threadIdx.x] = 0;
    t[threadIdx.x] = 0;

    __syncthreads();

    if (threadIdx.y == 0) {
        queue<COMP_COL_TYPE> in_queue(in_queue_[threadIdx.x], h + threadIdx.x , t + threadIdx.x, queue_size);
        uint8_t* chunk_ptr = (comp_ptr +  blk_offset_ptr[blockIdx.x ]);
        decompress_input< COMP_COL_TYPE> d(chunk_ptr, col_len);
        reader_warp<COMP_COL_TYPE>(d, in_queue);
    }

    else if (threadIdx.y == 1) {
    
        queue<COMP_COL_TYPE> in_queue(in_queue_[threadIdx.x], h + threadIdx.x, t + threadIdx.x, queue_size);
        //queue<decomp_write_queue_ele<DATA_TYPE>> out_queue(out_queue_[threadIdx.x], out_h + threadIdx.x, out_t + threadIdx.x, queue_size);
        input_stream<COMP_COL_TYPE, 2> s(&in_queue, (uint32_t)col_len, local_queue[threadIdx.x]);
        //decompress_output<DATA_TYPE> d((out + CHUNK_SIZE * (blockIdx.x )));
        //decoder_warp<COMP_COL_TYPE, DATA_TYPE, 8>(s, out_queue, d, CHUNK_SIZE, (DATA_TYPE*)(out + CHUNK_SIZE * blockIdx.x), COMP_COL_LEN);
        decoder_warp<COMP_COL_TYPE, DATA_TYPE, UDATA_TYPE,  2>(s, (DATA_TYPE *) out, (DATA_TYPE*)(out + CHUNK_SIZE * blockIdx.x), COMP_COL_LEN, READ_UNIT, col_len_ptr, CHUNK_SIZE);

    }

    __syncthreads();
}





template <typename COMP_COL_TYPE, typename DATA_TYPE, typename UDATA_TYPE, typename OUT_COL_TYPE, uint16_t queue_size = 4>
__global__ void 
//__launch_bounds__ (96, 13)
//__launch_bounds__ (128, 11)
inflate_orig(uint8_t* comp_ptr, uint8_t* out, const uint64_t* const col_len_ptr, const uint64_t* const blk_offset_ptr,  int COMP_COL_LEN, int READ_UNIT, uint64_t CHUNK_SIZE) {

    __shared__ COMP_COL_TYPE in_queue_[32][queue_size];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> h[32];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> t[32];


    __shared__ COMP_COL_TYPE local_queue[32][2];

    uint64_t col_len = (col_len_ptr[32 * (blockIdx.x) + threadIdx.x]);

    h[threadIdx.x] = 0;
    t[threadIdx.x] = 0;

    __syncthreads();

    if (threadIdx.y == 0) {
        queue<COMP_COL_TYPE> in_queue(in_queue_[threadIdx.x], h + threadIdx.x , t + threadIdx.x, queue_size);
        uint8_t* chunk_ptr = (comp_ptr +  blk_offset_ptr[blockIdx.x ]);
        decompress_input< COMP_COL_TYPE> d(chunk_ptr, col_len);
        reader_warp<COMP_COL_TYPE>(d, in_queue);
    }

    else if (threadIdx.y == 1) {
    
        queue<COMP_COL_TYPE> in_queue(in_queue_[threadIdx.x], h + threadIdx.x, t + threadIdx.x, queue_size);
        //queue<decomp_write_queue_ele<DATA_TYPE>> out_queue(out_queue_[threadIdx.x], out_h + threadIdx.x, out_t + threadIdx.x, queue_size);
        input_stream<COMP_COL_TYPE, 2> s(&in_queue, (uint32_t)col_len, local_queue[threadIdx.x]);
        //decompress_output<DATA_TYPE> d((out + CHUNK_SIZE * (blockIdx.x )));
        //decoder_warp<COMP_COL_TYPE, DATA_TYPE, 8>(s, out_queue, d, CHUNK_SIZE, (DATA_TYPE*)(out + CHUNK_SIZE * blockIdx.x), COMP_COL_LEN);
        decoder_warp<COMP_COL_TYPE, DATA_TYPE, UDATA_TYPE,  2>(s, (DATA_TYPE *) out, (DATA_TYPE*)(out + CHUNK_SIZE * blockIdx.x), COMP_COL_LEN, READ_UNIT, col_len_ptr, CHUNK_SIZE);

    }

    __syncthreads();
}


template <typename COMP_COL_TYPE, typename DATA_TYPE, typename UDATA_TYPE, typename OUT_COL_TYPE, int NUM_CHUNKS, uint16_t queue_size = 4, int NT = 64, int BT = 32>
__global__ void 
//__launch_bounds__ (NT, BT)
__launch_bounds__ (NT)
//inflate_orig_dw(uint8_t* comp_ptr, uint8_t* out, const uint64_t* const col_len_ptr, const uint64_t* const blk_offset_ptr, uint64_t CHUNK_SIZE, int COMP_COL_LEN, uint64_t num_chunks) {
inflate_orig_dw(uint8_t* comp_ptr, uint8_t* out, const uint64_t* const col_len_ptr, const uint64_t* const blk_offset_ptr,  int COMP_COL_LEN, int READ_UNIT, uint64_t CHUNK_SIZE, uint64_t num_chunks) {


    __shared__ COMP_COL_TYPE in_queue_[NUM_CHUNKS][queue_size];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> h[32];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> t[32];

    __shared__ COMP_COL_TYPE local_queue[NUM_CHUNKS][2];

    //uint64_t col_len = (col_len_ptr[32 * (blockIdx.x) + threadIdx.x]);

    h[threadIdx.x] = 0;
    t[threadIdx.x] = 0;

    //if(blockIdx.x == 0 && threadIdx.x == 0) printf("NUM_CHUNKS: %i\n", NUM_CHUNKS);

    __syncthreads();
    uint8_t active_chunks = NUM_CHUNKS;
    if((blockIdx.x+1) * NUM_CHUNKS > num_chunks){
       active_chunks = num_chunks - blockIdx.x * NUM_CHUNKS;
    }

    int my_block_idx = 0;
    uint64_t col_len = 0;
    int my_queue = 0;


    if (threadIdx.y == 0) {
        
        my_queue =  threadIdx.x % NUM_CHUNKS;
        my_block_idx = blockIdx.x * NUM_CHUNKS + threadIdx.x % NUM_CHUNKS;
        //col_len = (blk_offset_ptr[my_block_idx+1] - blk_offset_ptr[my_block_idx]);
        col_len = (col_len_ptr[my_block_idx]+3)/4 * 4;
        queue<COMP_COL_TYPE> in_queue(in_queue_[my_queue], h + my_queue , t + my_queue, queue_size);
        decompress_input< COMP_COL_TYPE> d(comp_ptr, col_len, blk_offset_ptr[my_block_idx] / sizeof(COMP_COL_TYPE));
        //if(NUM_CHUNKS== 2 && (threadIdx.x == 0) || (threadIdx.x == 1)) printf("bid: %i read col len: %llu\n", blockIdx.x, col_len);

        reader_warp_orig< COMP_COL_TYPE, NUM_CHUNKS>(d, in_queue, active_chunks);
    }

    else {
           
        my_queue = (threadIdx.y - 1);
        my_block_idx =  (blockIdx.x * NUM_CHUNKS + threadIdx.y -1);
        col_len = col_len_ptr[my_block_idx];
        queue<COMP_COL_TYPE> in_queue(in_queue_[my_queue], h + my_queue, t + my_queue, queue_size);
        uint64_t chunk_id = blockIdx.x * NUM_CHUNKS + (threadIdx.y - 1);

        //if(NUM_CHUNKS== 2 && (threadIdx.x == 0) || (threadIdx.x == 1)) printf("bid: %i decode col len: %llu\n", blockIdx.x, col_len);

        input_stream<COMP_COL_TYPE, 2> s(&in_queue, (uint32_t)col_len, local_queue[my_queue], threadIdx.x == 0);
        DATA_TYPE* out_8B = (DATA_TYPE*)out + ((uint64_t)chunk_id * (CHUNK_SIZE / sizeof(DATA_TYPE) )) ;

       // decoder_warp_orig_dw<COMP_COL_TYPE, DATA_TYPE, UDATA_TYPE,  NUM_CHUNKS, 2>(s, (DATA_TYPE*)(out + chunk_id * CHUNK_SIZE / sizeof(DATA_TYPE)), COMP_COL_LEN, READ_UNIT, col_len_ptr, CHUNK_SIZE);

        decoder_warp_orig_dw<COMP_COL_TYPE, DATA_TYPE, UDATA_TYPE,  NUM_CHUNKS, 2>(s, (DATA_TYPE*)(out_8B), COMP_COL_LEN, READ_UNIT, col_len_ptr, CHUNK_SIZE, chunk_id);

    }

    __syncthreads();
}


template <typename COMP_COL_TYPE, typename DATA_TYPE, typename UDATA_TYPE, typename OUT_COL_TYPE, int NUM_CHUNKS, uint16_t queue_size = 4, int NT = 32, int BT = 32>
__global__ void 
//__launch_bounds__ (NT, BT)
__launch_bounds__ (NT)
inflate_orig_rdw(uint8_t* comp_ptr, uint8_t* out, const uint64_t* const col_len_ptr, const uint64_t* const blk_offset_ptr,  int COMP_COL_LEN, int READ_UNIT, uint64_t CHUNK_SIZE, uint64_t num_chunks) {

    __shared__ COMP_COL_TYPE in_queue_[NUM_CHUNKS][queue_size];

    __syncthreads();
    uint8_t active_chunks = NUM_CHUNKS;
    if((blockIdx.x+1) * NUM_CHUNKS > num_chunks){
       active_chunks = num_chunks - blockIdx.x * NUM_CHUNKS;
    }


    int   my_queue = (threadIdx.y);
    int   my_block_idx =  (blockIdx.x * NUM_CHUNKS + threadIdx.y);
    uint64_t    col_len = col_len_ptr[my_block_idx];
    //col_len = (col_len_ptr[my_block_idx]+3)/4 * 4;

    //if(blockIdx.x == 0 && ((threadIdx.x == 0) || (threadIdx.x == 1))) printf("bid: %i  col len: %llu\n", blockIdx.x, col_len);

    __syncthreads();

    full_warp_input_stream<COMP_COL_TYPE, queue_size> s(comp_ptr, col_len, blk_offset_ptr[my_block_idx] / sizeof(COMP_COL_TYPE), in_queue_[my_queue]);

    __syncthreads();
    uint64_t chunk_id = blockIdx.x * NUM_CHUNKS + (threadIdx.y);
    DATA_TYPE* out_8B = (DATA_TYPE*)out + ((uint64_t)chunk_id * (CHUNK_SIZE / sizeof(DATA_TYPE) )) ;
    //decoder_warp_orig_rdw<COMP_COL_TYPE, DATA_TYPE, UDATA_TYPE,  queue_size>(s, (DATA_TYPE*)(out), COMP_COL_LEN, READ_UNIT, col_len_ptr, CHUNK_SIZE);
    decoder_warp_orig_rdw<COMP_COL_TYPE, DATA_TYPE, UDATA_TYPE,  NUM_CHUNKS, queue_size>(s, (DATA_TYPE*)(out_8B), COMP_COL_LEN, READ_UNIT, col_len_ptr, CHUNK_SIZE, chunk_id);
    __syncthreads();

}



