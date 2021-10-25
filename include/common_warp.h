// #ifndef __ZLIB_H__
// #define __ZLIB_H__

#include <common.h>
#include <fstream>
#include <sys/types.h> 
#include <sys/stat.h> 
#include <fcntl.h>
#include <sys/mman.h>
#include <simt/atomic>
#include <iostream>

#define BUFF_LEN 2


#define UNCOMP 0
#define STATIC 1
#define DYNAMIC 2
#define FULL_MASK 0xFFFFFFFF

#define MASK_4_1  0x000000FF
#define MASK_4_2  0x0000FF00
#define MASK_4_3  0x00FF0000
#define MASK_4_4  0xFF000000

#define MASK_8_1  0x0000000F
#define MASK_8_2  0x000000F0
#define MASK_8_3  0x00000F00
#define MASK_8_4  0x0000F000
#define MASK_8_5  0x000F0000
#define MASK_8_6  0x00F00000
#define MASK_8_7  0x0F000000
#define MASK_8_8  0xF0000000

#define MAXBITS 15
#define FIXLCODES 288
#define MAXDCODES 30
#define MAXCODES 316



template <typename T>
struct queue {
    T* queue_;
    simt::atomic<uint8_t, simt::thread_scope_block>* head;
    simt::atomic<uint8_t, simt::thread_scope_block>* tail;
    uint8_t len;

    __device__
    queue(T* buff, simt::atomic<uint8_t, simt::thread_scope_block>* h, simt::atomic<uint8_t, simt::thread_scope_block>* t, const uint8_t l) {
        queue_ = buff;
        head = h;
        tail = t;
        len = l;

    }

    __device__
    void enqueue(const T* v) {

        const auto cur_tail = tail->load(simt::memory_order_relaxed);
        const auto next_tail = (cur_tail + 1) % len;



        while (next_tail == head->load(simt::memory_order_acquire))
            __nanosleep(50);


        queue_[cur_tail] = *v;
        tail->store(next_tail, simt::memory_order_release);


    }

    __device__
    void dequeue(T* v) {

        const auto cur_head = head->load(simt::memory_order_relaxed);
        while (cur_head == tail->load(simt::memory_order_acquire))
            __nanosleep(50);

        *v = queue_[cur_head];

        const auto next_head = (cur_head + 1) % len;

        head->store(next_head, simt::memory_order_release);

    }

    __device__
    void attempt_dequeue(T* v, bool* p) {
        const auto cur_head = head->load(simt::memory_order_relaxed);
        if (cur_head == tail->load(simt::memory_order_acquire)) {
            *p = false;
            return;
        }


        *v = queue_[cur_head];
        *p = true;

        const auto next_head = (cur_head + 1) % len;

        head->store(next_head, simt::memory_order_release);

    }
    
};



template <typename T, typename QUEUE_TYPE>
__forceinline__
__device__
 void comp_enqueue(T* v,  queue<QUEUE_TYPE>* q){
    T temp_v = *v;

    //#pragma unroll
    for(int i = 0; i <  (sizeof(T) / sizeof(QUEUE_TYPE)); i++){

          const auto cur_tail = q-> tail->load(simt::memory_order_relaxed);
          const auto next_tail = (cur_tail + 1) % ( q->len);

            while (next_tail ==  q-> head->load(simt::memory_order_acquire))
                __nanosleep(20);

           // q->queue_[cur_tail] = (QUEUE_TYPE) (((temp_v >> (i * 8 * sizeof(QUEUE_TYPE))) & (0x0FFFFFFFF)));
           // q->queue_[cur_tail] = (QUEUE_TYPE) (((temp_v >> (i * 8 * sizeof(QUEUE_TYPE))) ));
            
             if(i == 0)
                q->queue_[cur_tail] = (temp_v).x;
             else if(i == 1)
                q->queue_[cur_tail] = (temp_v).y;
             else if(i == 2)
                q->queue_[cur_tail] = (temp_v).z;
             else if(i == 3)
                q->queue_[cur_tail] = (temp_v).w;

            q->tail->store(next_tail, simt::memory_order_release);
      }
}


template <typename T, typename QUEUE_TYPE>
__forceinline__ __device__
 void warp_enqueue (T* v,  queue<QUEUE_TYPE>* q, uint8_t subchunk_idx, uint8_t enq_num){

    T my_v = *v;

    for(uint8_t i = 0; i < enq_num; i++){
        T cur_v = __shfl_sync(FULL_MASK, my_v, i);
        if(threadIdx.x == subchunk_idx){

            const auto cur_tail = q-> tail->load(simt::memory_order_relaxed);
            const auto next_tail = (cur_tail + 1) % ( q->len);

            while (next_tail ==  q-> head->load(simt::memory_order_acquire)){
                __nanosleep(20);
            }
            //printf("cur_v: %lx\n", cur_v);
            q -> queue_[cur_tail] = cur_v;
            q->tail->store(next_tail, simt::memory_order_release);
        }
        __syncwarp(FULL_MASK);

    }
}





//producer of input queue
template <typename READ_COL_TYPE, typename COMP_COL_TYPE >
struct decompress_input {


    uint16_t row_offset;
    uint16_t len;
    uint16_t read_bytes;
    uint32_t t_read_mask;
    uint64_t pointer_off;

    COMP_COL_TYPE* pointer;

    __device__
    decompress_input(const uint8_t* ptr, const uint16_t l, const uint64_t p_off) :
        pointer((COMP_COL_TYPE*) ptr), len(l), pointer_off(p_off) {
        t_read_mask = (0xffffffff >> (32 - threadIdx.x));
        row_offset = 0;
        read_bytes = 0;
    }



    __forceinline__
    __device__
    int8_t comp_read_data(const uint32_t alivemask, COMP_COL_TYPE* v) {

        int8_t read_count  = 0;
        bool read = (read_bytes) < len;
        uint32_t read_sync = __ballot_sync(alivemask, read);
      
        if (__builtin_expect (read_sync == 0, 0)){
            return -1;
        }
        
        if(read){
            *v = pointer[row_offset + __popc(read_sync & t_read_mask)];
            row_offset += __popc(read_sync);
            read_bytes += sizeof(COMP_COL_TYPE);
            read_count = sizeof(COMP_COL_TYPE);
        }

        __syncwarp(alivemask);

        return read_count;

    }



};



template <typename READ_COL_TYPE, typename COMP_COL_TYPE >
    __forceinline__
    __device__
    uint8_t comp_read_data_seq(const uint32_t alivemask, COMP_COL_TYPE* v, decompress_input<READ_COL_TYPE, COMP_COL_TYPE>& in, uint8_t src_idx) {

        uint16_t src_len = __shfl_sync(alivemask, in.len, src_idx) / sizeof(COMP_COL_TYPE);
        uint16_t src_read_bytes = __shfl_sync(alivemask, in.read_bytes, src_idx);
        uint64_t src_pointer_off = __shfl_sync(alivemask, in.pointer_off, src_idx);

        bool read = (src_read_bytes + threadIdx.x) < src_len;
        uint32_t read_sync = __ballot_sync(alivemask, read);



        if(read){
             uint64_t idx = src_read_bytes + threadIdx.x + src_pointer_off;

            // printf("tid; %i idx: %llu\n", threadIdx.x, idx);
            *v = in.pointer[src_read_bytes + threadIdx.x + src_pointer_off];
            //if(src_idx == 1 || blockIdx.x == 1)
             //   printf("tid: %i v: %lx\n", threadIdx.x, (*v));

        
        }

        uint8_t read_count = __popc(read_sync);
        
        if(threadIdx.x == src_idx){
            //printf("rb: %llu src len: %llu v: %llu \n", (unsigned long long)  in.read_bytes, (unsigned long long) src_len, *v);
            in.read_bytes += read_count;
        }

        return read_count;


    }


//consumer of input queue
template <typename READ_COL_TYPE, uint8_t buff_len = 4>
struct input_stream {


    union buff{
        READ_COL_TYPE* b;
        uint32_t* u;
    }b;
    uint8_t head;
    uint8_t count;
    uint8_t uint_head;
    uint8_t uint_count;
    
    
    queue<READ_COL_TYPE>* q;
    uint32_t read_bytes;
    uint32_t expected_bytes;
    uint8_t bu_size = (sizeof(READ_COL_TYPE)* buff_len)/sizeof(uint32_t);

    uint8_t uint_bit_offset;

    __device__
    input_stream(queue<READ_COL_TYPE>* q_, uint32_t eb, READ_COL_TYPE* shared_b, bool pass) {
    //input_stream(queue<READ_COL_TYPE>* q_, uint32_t eb) {
        if(pass){
        q = q_;
        expected_bytes = eb;

        b.b = shared_b;

        head = 0;

        uint_bit_offset = 0;
        uint_count = 0;
        uint_head = 0;
        read_bytes = 0;
        count = 0;
        for (; (count < buff_len) && (read_bytes < expected_bytes);
             count++, read_bytes += sizeof(READ_COL_TYPE), uint_count += sizeof(READ_COL_TYPE)/sizeof(uint32_t)) {
            // q->dequeue(b.b + count);
            q -> dequeue(b.b + count);

        }
        }
    }

    template<typename T>
    __device__
    void get_n_bits(const uint32_t n, T* out) {

        *out = (T) 0;

        ((uint32_t*) out)[0] = __funnelshift_rc(b.u[(uint_head)], b.u[(uint_head+1)%bu_size], uint_bit_offset);

        if (32 > n) {
            ((uint32_t*) out)[0] <<= (32 - n);
            ((uint32_t*) out)[0] >>= (32 - n);
        }


        uint_bit_offset += n;
        if (uint_bit_offset >= 32) {
            uint_bit_offset = uint_bit_offset % 32;
            uint_head = (uint_head+1) % bu_size;
            if ((uint_head % (sizeof(READ_COL_TYPE)/sizeof(uint32_t))) == 0) {
                head = (head + 1) % buff_len;
                count--;
            }

            uint_count--;

        }
  

    }
    //T should be at least 32bits
    template<typename T>
    __device__
    void fetch_n_bits(const uint32_t n, T* out) {
        while ((count < buff_len) && (read_bytes < expected_bytes)) {
            q -> dequeue(b.b + ((head+count) % buff_len));
           
            count++;
            uint_count += sizeof(READ_COL_TYPE)/sizeof(uint32_t);
            read_bytes += sizeof(READ_COL_TYPE);
        }

        get_n_bits<T>(n, out);
    }

    __device__
    void align_bits(){
        if(uint_bit_offset % 8 != 0){
            uint_bit_offset = ((uint_bit_offset + 7)/8) * 8;
            if(uint_bit_offset == 32){
                uint_bit_offset = 0;
                uint_head = (uint_head+1) % bu_size;
                if ((uint_head % (sizeof(READ_COL_TYPE)/sizeof(uint32_t))) == 0) {
                    head = (head + 1) % buff_len;
                    count--;
                }
                uint_count--;
            }
        }
      
    }

    template<typename T>
    __device__
    void peek_n_bits(const uint32_t n, T* out) {
        while ((count < buff_len) && (read_bytes < expected_bytes)) { 
            //q->dequeue(b.b + ((head+count) % buff_len));
            q -> dequeue(b.b + ((head+count) % buff_len));

            count++;
            uint_count += sizeof(READ_COL_TYPE)/sizeof(uint32_t);
            read_bytes += sizeof(READ_COL_TYPE);
        }
        uint8_t count_ = count;
        uint8_t head_ = head;
        uint8_t uint_count_ = uint_count;
        uint8_t uint_head_ = uint_head;
        uint8_t uint_bit_offset_ = uint_bit_offset;

        get_n_bits<T>(n, out);

        count = count_;
        head = head_;
        uint_count = uint_count_;
        uint_head = uint_head_;
        uint_bit_offset = uint_bit_offset_;
    }


};




//#endif // __ZLIB_H__
