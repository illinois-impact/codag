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



__constant__ int32_t fixed_lengths[FIXLCODES];


struct dynamic_huffman {
    int16_t lensym[FIXLCODES];
    int16_t treelen[MAXCODES];
    //int16_t off[MAXBITS + 1];
}; //__attribute__((aligned(16)));


struct fix_huffman {
    int16_t lencnt[MAXBITS + 1];
    int16_t lensym[FIXLCODES];
    int16_t distcnt[MAXBITS + 1];
    int16_t distsym[MAXDCODES];
};// __attribute__((aligned(16)));

typedef struct __align__(32)
{
    simt::atomic<uint8_t, simt::thread_scope_device>  counter;
    simt::atomic<uint8_t, simt::thread_scope_device>  lock[32];

} __attribute__((aligned (32))) slot_struct;



struct  write_queue_ele{
    uint32_t data;
    uint8_t type;
};


//__forceinline__
 __device__ unsigned warp_id()
{
    // this is not equal to threadIdx.x / 32
    unsigned ret;
    asm volatile ("mov.u32 %0, %warpid;" : "=r"(ret));
    return ret;
}
//__forceinline__ 
__device__ uint32_t get_smid() {
     uint32_t ret;
     asm  ("mov.u32 %0, %smid;" : "=r"(ret) );
     return ret;
}

//__forceinline__
__device__ uint8_t find_slot(uint32_t sm_id, slot_struct* slot_array){

    //bool fail = true;
    //uint64_t count = 0;
    uint8_t page = 0;

    do{
        page = ((slot_array[sm_id]).counter.fetch_add(1, simt::memory_order_acquire)) % 32;

        bool lock = false;
        uint8_t v = (slot_array[sm_id]).lock[page].load(simt::memory_order_acquire);
        if (v == 0)
        {
            lock = (slot_array[sm_id]).lock[page].compare_exchange_strong(v, 1, simt::memory_order_acquire, simt::memory_order_relaxed);

            if(lock){
                //(slot_array[sm_id]).lock[page].store(0, simt::memory_order_release);
                return page;
            }
        }

    } while(true);

    return page;
}

__forceinline__
__device__ void release_slot(uint32_t sm_id, uint32_t page, slot_struct* slot_array){
    (slot_array[sm_id]).lock[page].store(0, simt::memory_order_release);
}


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

    // __device__
    // void enqueue(const T* v) {

    //     const auto cur_tail = tail->load(simt::memory_order_relaxed);
    //     const auto next_tail = (cur_tail + 1) % len;



    //     while (next_tail == head->load(simt::memory_order_acquire))
    //         __nanosleep(50);


    //     queue_[cur_tail] = *v;
    //     tail->store(next_tail, simt::memory_order_release);


    // }

    // __device__
    // void dequeue(T* v) {

    //     const auto cur_head = head->load(simt::memory_order_relaxed);
    //     while (cur_head == tail->load(simt::memory_order_acquire))
    //         __nanosleep(50);

    //     *v = queue_[cur_head];

    //     const auto next_head = (cur_head + 1) % len;

    //     head->store(next_head, simt::memory_order_release);

    // }

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

template <typename T>
__forceinline__
__device__
 void g_enqueue(T* v,  queue<T>* q){
      const auto cur_tail = q-> tail->load(simt::memory_order_relaxed);
        const auto next_tail = (cur_tail + 1) % ( q->len);

        while (next_tail ==  q-> head->load(simt::memory_order_acquire))
            __nanosleep(20);

         q->queue_[cur_tail] = *v;
         q->tail->store(next_tail, simt::memory_order_release);
}


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




template <typename T>
__forceinline__
__device__
 void g_dequeue(T* v,  queue<T>* q){
       const auto cur_head = q -> head->load(simt::memory_order_relaxed);
        while (cur_head ==  q->tail->load(simt::memory_order_acquire))
            __nanosleep(20);

        *v =  (q->queue_)[cur_head];

        const auto next_head = (cur_head + 1) % (q->len);
        q->head->store(next_head, simt::memory_order_release);
}



//producer of input queue
template <typename READ_COL_TYPE, typename COMP_COL_TYPE >
struct decompress_input {


    //uint64_t col_width = sizeof(READ_COL_TYPE);
    uint16_t row_offset;
    uint16_t len;
    uint16_t read_bytes;
    uint32_t t_read_mask;
   // READ_COL_TYPE* pointer;

    COMP_COL_TYPE* pointer;


    //READ_COL_TYPE in_buff[IN_BUFF_LEN];

    __device__
    decompress_input(const uint8_t* ptr, const uint16_t l) :
        pointer((COMP_COL_TYPE*) ptr), len(l) {
        //uint8_t tid = threadIdx.x;
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
           // read_count = -1;
            return -1;
        }
        
        //if (__builtin_expect (read, 1)) {
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
    int8_t comp_read_data2(const uint32_t alivemask, COMP_COL_TYPE* v, decompress_input<READ_COL_TYPE, COMP_COL_TYPE>& in) {

        int8_t read_count  = 0;
        bool read = (in.read_bytes) < in.len;
        uint32_t read_sync = __ballot_sync(alivemask, read);
      
        if (read_sync == 0)
            read_count = -1;
        
        //if (__builtin_expect (read, 1)) {
        if(read){
            *v = in.pointer[in.row_offset + __popc(read_sync & in.t_read_mask)];
            in.row_offset += __popc(read_sync);
            in.read_bytes += sizeof(COMP_COL_TYPE);
            read_count = sizeof(COMP_COL_TYPE);
        }



        __syncwarp(alivemask);

        return read_count;

    }


//consumer of input queue
template <typename READ_COL_TYPE, uint8_t buff_len = 4>
struct input_stream {

    // union buff {
    //     READ_COL_TYPE b[buff_len];
    //     uint32_t u[(sizeof(READ_COL_TYPE)* buff_len)/sizeof(uint32_t)];
    // } b;
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
    input_stream(queue<READ_COL_TYPE>* q_, uint32_t eb, READ_COL_TYPE* shared_b) {
    //input_stream(queue<READ_COL_TYPE>* q_, uint32_t eb) {

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
            g_dequeue<READ_COL_TYPE>(b.b + count, q);

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
    template<typename T>
    __device__
    void fetch_n_bits(const uint32_t n, T* out) {

        while ((count < buff_len) && (read_bytes < expected_bytes)) {
            //q->dequeue(b.b + ((head+count) % buff_len));
            g_dequeue<READ_COL_TYPE>(b.b + ((head+count) % buff_len), q);

            //q->dequeue(s_b[((head+count) % buff_len)]);
           
            count++;
            uint_count += sizeof(READ_COL_TYPE)/sizeof(uint32_t);
            read_bytes += sizeof(READ_COL_TYPE);
        }

        get_n_bits<T>(n, out);
    }

    template<typename T>
    __device__
    void peek_n_bits(const uint32_t n, T* out) {
        while ((count < buff_len) && (read_bytes < expected_bytes)) { 
            //q->dequeue(b.b + ((head+count) % buff_len));
            g_dequeue<READ_COL_TYPE>(b.b + ((head+count) % buff_len), q);

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



// void writer_warp(queue<uint64_t>& mq, decompress_output<size_t WRITE_COL_LEN = 512>& out) {
template <size_t WRITE_COL_LEN = 512, size_t CHUNK_SIZE = 8192>
struct decompress_output {

    uint8_t* out_ptr;
    uint16_t counter;
    uint16_t total;

    __device__
    decompress_output(uint8_t* ptr):
        out_ptr(ptr) {
            counter = 0;
            total = CHUNK_SIZE / 32;
    }



    __forceinline__ __device__
    void write_byte(uint64_t write_counter, uint8_t b, uint32_t t_off){
        out_ptr[write_counter + (write_counter/WRITE_COL_LEN) * (31*WRITE_COL_LEN) + t_off] = b;
    }

    __forceinline__ __device__
    void copy_byte(uint16_t read_counter, uint16_t write_counter, uint16_t t_off){
        out_ptr[write_counter + (write_counter/WRITE_COL_LEN) * (31*WRITE_COL_LEN) + t_off] =
            out_ptr[read_counter + (read_counter/WRITE_COL_LEN) * (31*WRITE_COL_LEN) + t_off];
    }

    //   __forceinline__ __device__
    // void copy_byte2(uint64_t read_counter, uint64_t write_counter, uint32_t t_off){
    //     out_ptr[write_counter + (write_counter >> 9) * (31*WRITE_COL_LEN) + t_off] =
    //         out_ptr[read_counter + (read_counter >> 9) * (31*WRITE_COL_LEN) + t_off];
    // }

    // __forceinline__ __device__
    // void copy_8byte(uint64_t read_counter, uint64_t write_counter, uint32_t t_off) {
    //    int w_idx =  (write_counter + (write_counter/WRITE_COL_LEN) * (31*WRITE_COL_LEN) + t_off) / 8;
    //    int r_idx = (read_counter + (read_counter/WRITE_COL_LEN) * (31*WRITE_COL_LEN) + t_off) / 8;
    //    ((uint64_t*)out_ptr)[w_idx] = ((uint64_t*)out_ptr)[r_idx];

    // }

    // __forceinline__ __device__ 
    // void new_memcpy(uint64_t read_counter, uint64_t write_counter, uint32_t len, uint32_t offset, uint32_t t_off) {
    //     uint32_t write_len = len;
    //     uint64_t rc = read_counter;
    //     uint64_t wc = write_counter; 

    //     int slow_bytes = min(write_len, (int)((8 -(size_t)(wc)) & 0x7));
        
    //     //number of bytes to write per thread
    //     int slow_bytes_t = (max(slow_bytes - threadIdx.x, 0) + 31) / 32;

    //     if (slow_bytes){
    //         uint64_t temp_rc = rc + threadIdx.x;
    //         uint64_t temp_wc = wc + threadIdx.x;
    //         for (int i = 0; i < slow_bytes_t; i++){
    //             copy_byte(temp_rc, temp_wc, t_off);
    //             temp_rc += 32;
    //             temp_wc += 32;
    //         }
    //         write_len -= slow_bytes;
    //         wc += (slow_bytes);
    //         rc += (slow_bytes);
    //     }

    //     __syncwarp();
    //     int fast_bytes = write_len;
    //     int fast_remain = fast_bytes & (~(32 - 1));
    //     int fast_bytes_t = (max(fast_remain - threadIdx.x*8, 0) + 32*8 - 1) / (32*8);
    //     if(fast_remain > 0){
    //         fast_bytes_t = (fast_remain - threadIdx.x*8);
    //         if(fast_bytes_t < 0)
    //             fast_bytes_t = 0;

    //         fast_bytes_t = ((fast_remain - threadIdx.x*8) + 32*8 - 1) / (32*8);
    //     }

    //     //8byte copy
    //     if (fast_remain > 0) {
    //        // printf("tid: %i fast bytes:%i, slow_bytes:%i wc: %lu, len: %lu\n",threadIdx.x, fast_bytes, slow_bytes, (unsigned long) wc ,(unsigned long)len);
    //         uint64_t temp_rc = rc + threadIdx.x * 8;
    //         uint64_t temp_wc = wc + threadIdx.x * 8;
    //         for(int i = 0; i < fast_bytes_t; i++){
    //             copy_8byte(temp_rc,temp_wc,t_off);
    //             temp_rc += 8 * 32;
    //             temp_wc += 8 * 32;
    //         }
    //         write_len -= fast_remain;
    //         wc += fast_remain;
    //         rc += fast_remain;
    //     }
        
    //     slow_bytes = write_len;

    //     //slow_copy remainder
    //     if(write_len != 0){
    //         //printf("tid: %i write_len:%lu,  wc: %lu, len: %lu\n",threadIdx.x, (unsigned long)write_len, (unsigned long) wc ,(unsigned long)len);

    //         slow_bytes_t = (max(slow_bytes - threadIdx.x, 0) + 31) / 32;
    //         //printf("tid: %i write_len:%lu,  slow_bytes_t: %lu, len: %lu\n",threadIdx.x, (unsigned long)write_len, (unsigned long) slow_bytes_t ,(unsigned long)len);
    //             rc += threadIdx.x;
    //             wc += threadIdx.x;
    //           for (int i = 0; i < slow_bytes_t; i++){
    //             //printf("copy tid: %i write_len:%lu,  slow_bytes_t: %lu, len: %lu\n",threadIdx.x, (unsigned long)write_len, (unsigned long) slow_bytes_t ,(unsigned long)len);
           
    //             copy_byte(rc, wc, t_off);
    //             wc += 32;
    //             rc += 32;
    //         }
    //     }

    // }


    // __forceinline__ __device__
    // void col_memcpy(uint8_t idx, uint16_t len, uint16_t offset) {
      
    //     //copy the meta data for the ptr


    //     uint16_t orig_counter = __shfl_sync(FULL_MASK, counter, idx);
    //     uint16_t t_off = WRITE_COL_LEN * idx;

    //     uint8_t num_writes = ((len - threadIdx.x + 31) / 32);
        
    //     uint16_t start_counter =  0;
    //     if(orig_counter > offset)
    //         start_counter = orig_counter - offset;
    //     uint16_t read_counter = start_counter + threadIdx.x;
    //     uint16_t write_counter = orig_counter + threadIdx.x;

        
    //     // //if(offset > len){
    //     // if(1 == 0){
    //     //     new_memcpy(start_counter, orig_counter, len, offset, t_off);
    //     //    // __syncwarp();
    //     // }

    //     //else {
    //         #pragma unroll 
    //         for(uint8_t i = 0; i < num_writes; i++){

    //             //check offset
    //             if(read_counter >= orig_counter){
    //                 read_counter = (read_counter - orig_counter) % offset + start_counter;
    //             }
    //             //uint8_t read_byte = 0;
    //             copy_byte(read_counter, write_counter, t_off);
    //             read_counter += 32;
    //             write_counter += 32;
    //         }
    //     //}
    //     //set the counter
    //     if(threadIdx.x == idx)
    //         counter += len;
    // }
    
    template <uint32_t NUM_THREAD = 8>
    //__forceinline__ 
    __device__
    void col_memcpy_div(uint8_t idx, uint16_t len, uint16_t offset, uint8_t div, uint32_t MASK) {
      
        uint16_t t_off = WRITE_COL_LEN * idx;

        // if(len <= 4 && offset){
        //     if(threadIdx.x == idx){
        //        // printf("len2 write\n");
        //        // ((uint32_t*)out_ptr + ((counter + (counter/WRITE_COL_LEN) * (31*WRITE_COL_LEN) + t_off) /4))[0] =  ((uint32_t*)out_ptr + ((counter - offset + ((counter-offset)/WRITE_COL_LEN) * (31*WRITE_COL_LEN) + t_off) /4))[0];
        //        // uint32_t r_data = ((uint32_t*)out_ptr + ((counter - offset + ((counter-offset)/WRITE_COL_LEN) * (31*WRITE_COL_LEN) + t_off) /4))[0];
        //        // uint32_t r_data = ((uint32_t*)(out_ptr[counter - offset + ((counter - offset)/WRITE_COL_LEN) * (31*WRITE_COL_LEN) + WRITE_COL_LEN * idx]))[0];
        //         uint32_t r_data = 0;
        //         for(int i = len - 1; i >= 0; i--){
        //             out_ptr[counter + (counter/WRITE_COL_LEN) * (31*WRITE_COL_LEN) + WRITE_COL_LEN * idx] = (uint8_t)((r_data >> (i * 8)) & 0x0F);
        //             counter++;
        //         }
        //        // counter += len;
        //     }
        //     return;
        // }

        int tid = threadIdx.x - div * NUM_THREAD;
        uint16_t orig_counter = __shfl_sync(MASK, counter, idx);


        uint8_t num_writes = ((len - tid + NUM_THREAD - 1) / NUM_THREAD);
        
        uint16_t start_counter =  orig_counter - offset;
        //if(orig_counter > offset)
           // start_counter = orig_counter - offset;
        uint16_t read_counter = start_counter + tid;
        uint16_t write_counter = orig_counter + tid;

        if(read_counter >= orig_counter){
                read_counter = (read_counter - orig_counter) % offset + start_counter;
        }

        #pragma unroll 
        for(int i = 0; i < num_writes; i++){

            //check offset
            // if(read_counter >= orig_counter){
            //     read_counter = (read_counter - orig_counter) % offset + start_counter;
            // }

            //uint8_t read_byte = 0;
            //copy_byte(read_counter, write_counter, t_off);
                out_ptr[write_counter + (write_counter/WRITE_COL_LEN) * (31*WRITE_COL_LEN) + WRITE_COL_LEN * idx] =
            out_ptr[read_counter + (read_counter/WRITE_COL_LEN) * (31*WRITE_COL_LEN) + WRITE_COL_LEN * idx];

            //  out_ptr[write_counter + (write_counter >> 9) * (31*WRITE_COL_LEN) + WRITE_COL_LEN * idx] =
            // out_ptr[read_counter + (read_counter >> 9) * (31*WRITE_COL_LEN) + WRITE_COL_LEN * idx];

            read_counter += NUM_THREAD;
            write_counter += NUM_THREAD;
        }
    
        //set the counter
        if(threadIdx.x == idx)
            counter += len;

    }


    //__forceinline__ 
    __device__
    void write_literal(uint8_t idx, uint8_t b){
        if(threadIdx.x == idx){
            //write_byte(counter, b,  WRITE_COL_LEN * idx);
            out_ptr[counter + (counter/WRITE_COL_LEN) * (31*WRITE_COL_LEN) +   WRITE_COL_LEN * idx] = b;
            counter++;
        }
    }


};

template <typename READ_COL_TYPE, typename COMP_COL_TYPE>
//__forceinline__ 
__device__
void reader_warp(decompress_input<READ_COL_TYPE, COMP_COL_TYPE>& in, queue<READ_COL_TYPE>& rq) {
    while (true) {
        COMP_COL_TYPE v;
        int8_t rc = in.comp_read_data(FULL_MASK, &v);
        //int8_t rc = comp_read_data2(FULL_MASK, &v, in);

        if (rc == -1)
            break;
        else if (rc > 0){
            //rq.enqueue(&v);
            comp_enqueue<COMP_COL_TYPE, READ_COL_TYPE>(&v, &rq);
        }
    }
}


template <typename READ_COL_TYPE, size_t in_buff_len = 4>
//__forceinline__
 __device__
int16_t decode (input_stream<READ_COL_TYPE, in_buff_len>& __restrict__ s, const int16_t* const __restrict__  counts, const int16_t*  __restrict__ symbols){

    //unsigned int len;
    //unsigned int code;
    //unsigned int count;
    uint32_t next32r = 0;
    s.template peek_n_bits<uint32_t>(32, &next32r);
    //if(threadIdx.x == 1)
   // printf("next: %lx\n",(unsigned long)next32r );

    next32r = __brev(next32r);


    uint32_t first = 0;
    #pragma unroll
    for (uint8_t len = 1; len <= MAXBITS; len++) {
        uint32_t code  = (next32r >> (32 - len)) - first;
        
        uint16_t count = counts[len];
    if (code < count) 
    {
        uint32_t temp;
        s.template fetch_n_bits<uint32_t>(len, &temp);
        return symbols[code];
    }
        symbols += count;  
        first += count;
        first <<= 1;
    }

    return -10;
}



//Construct huffman tree
template <typename READ_COL_TYPE, size_t in_buff_len = 4>
//__forceinline__
 __device__ 
void construct(input_stream<READ_COL_TYPE, in_buff_len>& __restrict__ s, int16_t* const __restrict__ counts , int16_t* const  __restrict__ symbols, 
    const int16_t* const __restrict__ length,  int16_t* const __restrict__ offs, const int num_codes){


    int len;
    #pragma unroll
    for(len = 0; len <= MAXBITS; len++){
        counts[len] = 0;
    }

    for(len = 0; len < num_codes; len++){
        symbols[len] = 0;
        (counts[length[len]])++;
    }
  

    //int16_t offs[16];
    //offs[0] = 0;
    offs[1] = 0;

    for (len = 1; len < MAXBITS; len++){
        offs[len + 1] = offs[len] + counts[len];
    }

    for(int16_t symbol = 0; symbol < num_codes; symbol++){
         if (length[symbol] != 0){
            symbols[offs[length[symbol]]++] = symbol;
            //offs[length[symbol]]++;
        }
    }
        
}


// //Construct huffman tree
// template <typename READ_COL_TYPE, size_t in_buff_len = 4>
// __forceinline__ __device__ 
// void fix_construct(input_stream<READ_COL_TYPE, in_buff_len>& s, int16_t* counts, int16_t* symbols, const int32_t *length, int16_t* offs, int num_codes){

//     int symbol;
//     int len;
//     int left;

//     for(len = 0; len < num_codes; len++){
//         symbols[len] = 0;
//     }

//     for(len = 0; len <= MAXBITS; len++){
//         counts[len] = 0;
//     }

//     for(symbol = 0; symbol < num_codes; symbol++){
//         (counts[fixed_lengths[symbol]])++;
//     }

//     left = 1;
//     for(len = 1; len <= MAXBITS; len++){
//         left <<= 1;
//         left -= counts[len];       
//         if (left < 0) 
//             return; 
//     }

    
//         //computing offset array for conunts
//        // int16_t offs[MAXBITS + 1];
//         offs[0] = 0;
//         offs[1] = 0;
//         for (len = 1; len < MAXBITS; len++)
//             offs[len + 1] = offs[len] + counts[len];

//         for(symbol = 0; symbol < num_codes; symbol++){
//              if (fixed_lengths[symbol] != 0){
//                 symbols[offs[fixed_lengths[symbol]]] = symbol;
//                 offs[fixed_lengths[symbol]]++;
//             }
//         }
// }

/// permutation of code length codes
static const __device__ __constant__ uint8_t g_code_order[19 + 1] = {
  16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15, 0xff};


//construct huffman tree for dynamic huffman encoding block
template <typename READ_COL_TYPE, size_t in_buff_len = 4>
//__forceinline__
 __device__
//void decode_dynamic(input_stream<READ_COL_TYPE, in_buff_len>& s, dynamic_huffman* huff_tree_ptr, unsigned warp_id, unsigned sm_id, int32_t* d_lengths, int16_t* d_off){
void decode_dynamic(input_stream<READ_COL_TYPE, in_buff_len>& s, dynamic_huffman* huff_tree_ptr, const uint32_t buff_idx,
    int16_t* const s_len, int16_t* const s_distcnt, int16_t* const s_distsym, int16_t* const s_off){



    uint16_t hlit;
    uint16_t hdist;
    uint16_t hclen;

    //int16_t* offs = huff_tree_ptr[buff_idx].off;

    //getting the meta data for the compressed block

    uint32_t head_temp;
    s.template fetch_n_bits<uint32_t>(14, &head_temp);
    hlit = (head_temp & (0x1F));
    head_temp >>= 5;
    hdist = (head_temp & (0x1F));
    head_temp >>= 5;
    hclen = (head_temp);

    hlit += 257;
    hdist += 1;
    hclen += 4;
    //int32_t lengths[MAXCODES];
    int index = 1;


    //check
    uint32_t temp;
    s.template fetch_n_bits<uint32_t>(12, &temp);
    int16_t* lengths = huff_tree_ptr[buff_idx].treelen;

    for (index = 0; index < 4; index++) {
          lengths[g_code_order[index]] = (int16_t)(temp & 0x07);
            temp >>=3;
    }
   // #pragma unroll
    for (index = 4; index < hclen; index++) {
        s.template fetch_n_bits<uint32_t>(3, &temp);
        lengths[g_code_order[index]] = (int16_t)temp;
    }

    for (; index < 19; index++) {
        lengths[g_code_order[index]] = 0;
    }
       
    construct<READ_COL_TYPE, in_buff_len>(s, s_len, huff_tree_ptr[buff_idx].lensym, lengths, s_off, 19);


     index = 0;
     //symbol;
    while (index < hlit + hdist) {
        int32_t symbol =  decode<READ_COL_TYPE, in_buff_len>(s, s_len, huff_tree_ptr[buff_idx].lensym);

        //represent code lengths of 0 - 15
        if(symbol < 16){
            lengths[(index++)] = symbol;
        }

        else{
            int16_t len = 0;
            if(symbol == 16) {
                 len = lengths[index - 1];  // last length
                 s.template fetch_n_bits<int32_t>(2, &symbol);
                 symbol += 3;
            }
            else if(symbol == 17){
                s.template fetch_n_bits<int32_t>(3, &symbol);
                symbol += 3;
            }
            else if(symbol == 18) {
                s.template fetch_n_bits<int32_t>(7, &symbol);
                symbol += 11;
            }
       

            while(symbol-- > 0){
                lengths[index++] = len;
            }

        }
    }



    //construct<READ_COL_TYPE, in_buff_len>(s, huff_tree_ptr[buff_idx].lencnt, huff_tree_ptr[buff_idx].lensym, lengths, huff_tree_ptr[buff_idx].off, hlit);
    //construct<READ_COL_TYPE, in_buff_len>(s, s_len, huff_tree_ptr[buff_idx].lensym, lengths, huff_tree_ptr[buff_idx].off, hlit);
    //construct<READ_COL_TYPE, in_buff_len>(s, s_distcnt, s_distsym, &(lengths[hlit]), huff_tree_ptr[buff_idx].off, hdist);

    construct<READ_COL_TYPE, in_buff_len>(s, s_len, huff_tree_ptr[buff_idx].lensym, lengths, s_off, hlit);
    construct<READ_COL_TYPE, in_buff_len>(s, s_distcnt, s_distsym, (lengths + hlit), s_off, hdist);

   
    //printf("end\n" );

    return;
}


// template <typename READ_COL_TYPE, size_t in_buff_len = 4>
// __forceinline__ __device__ 
// void decode_fixed(input_stream<READ_COL_TYPE, in_buff_len>& s, dynamic_huffman* huff_tree_ptr, unsigned warp_id, unsigned sm_id){
//     //int32_t lengths[MAXCODES];
//     // unsigned buff_idx = (sm_id * 64 + warp_id)*32 + threadIdx.x;
//     // //unsigned buff_idx = 0;
//     // // int32_t* lengths = &(d_lengths[buff_idx * MAXCODES]);
//     // // int16_t* offs = &(d_off[buff_idx * (MAXBITS + 1)]);


//     // int16_t* offs = huff_tree_ptr[buff_idx].off;
//     // int16_t* lengths = huff_tree_ptr[buff_idx].treelen;


//     // int symbol;

//     // for (symbol = 0; symbol < 144; symbol++) lengths[symbol] = 8;
//     // for (; symbol < 256; symbol++) lengths[symbol] = 9;
//     // for (; symbol < 280; symbol++) lengths[symbol] = 7;
//     // for (; symbol < FIXLCODES; symbol++) lengths[symbol] = 8;
    



//     // fix_construct<READ_COL_TYPE, in_buff_len>(s, huff_tree_ptr[(sm_id * 64 + warp_id)*32 + threadIdx.x].lencnt,  huff_tree_ptr[(sm_id * 64 + warp_id)*32 + threadIdx.x].lensym, lengths, offs, FIXLCODES);
    
//     // for (symbol = 0; symbol < MAXDCODES; symbol++) lengths[symbol] = 5;
 
//     // construct<READ_COL_TYPE, in_buff_len>(s, huff_tree_ptr[(sm_id * 64 + warp_id)*32 + threadIdx.x].distcnt,  huff_tree_ptr[(sm_id * 64 + warp_id)*32 + threadIdx.x].distsym, lengths, offs, MAXDCODES);

//     return;
// }



//code starts from 257
static const __device__ __constant__ uint16_t g_lens[29] = {  // Size base for length codes 257..285
  3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
  31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};

//code starts from 257
static const __device__ __constant__ uint16_t
  g_lext[29] = { 
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};


static const __device__ __constant__ uint16_t
  g_dists[30] = {  // Offset base for distance codes 0..29
    1,   2,   3,   4,   5,   7,    9,    13,   17,   25,   33,   49,   65,    97,    129,
    193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};

static const __device__ __constant__ uint16_t g_dext[30] = {  // Extra bits for distance codes 0..29
  0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

// template <typename READ_COL_TYPE, size_t in_buff_len = 4>
// __forceinline__ __device__ 
// void decode_symbol(input_stream<READ_COL_TYPE, in_buff_len>& s, queue<write_queue_ele>& mq, const fix_huffman* const huff_tree_ptr, unsigned buff_idx) {


//     while(1){
              
//         uint16_t sym = decode<READ_COL_TYPE, in_buff_len>(s,  huff_tree_ptr[buff_idx].lencnt,  huff_tree_ptr[buff_idx].lensym);


//         //parse 5 bits
//         //not compressed, literal
//         if(sym <= 255) {
//             write_queue_ele qe;
//             qe.type = 0;
//             qe.data = (uint32_t) sym;
//             //mq.enqueue(&qe);
//             g_enqueue<write_queue_ele>(&qe, &mq);
                        
//         }

//         //end of block
//         else if(sym == 256) {
//             break;
//         }

//         //lenght, need to parse

//         else{

//            // uint16_t extra_bits = g_lext[sym - 257];

//             uint32_t extra_len  = 0;
//             if( g_lext[sym - 257] != 0){
//                s.template fetch_n_bits<uint32_t>( g_lext[sym - 257], &extra_len);
//             }

//             uint16_t len = extra_len + g_lens[sym - 257];
//             //distance, 5bits
//             uint16_t sym_dist = decode<READ_COL_TYPE, in_buff_len>(s,  huff_tree_ptr[buff_idx].distcnt, huff_tree_ptr[buff_idx].distsym);    
//             //uint16_t extra_bits_dist = g_dext[sym_dist];
            
//             uint32_t extra_len_dist = 0;
//             if(g_dext[sym_dist] != 0){
//                 s.template fetch_n_bits<uint32_t>(g_dext[sym_dist], &extra_len_dist);
//             }


//            //uint16_t dist = extra_len_dist + g_dists[sym_dist];

//             write_queue_ele qe;
//             qe.data = (len << 16) | (extra_len_dist + g_dists[sym_dist]);
//             qe.type = 1;
//             //mq.enqueue(&qe);
//             g_enqueue<write_queue_ele>(&qe, &mq);

//         }
//     }

// }


template <typename READ_COL_TYPE, size_t in_buff_len = 4>
//__forceinline__ 
__device__ 
void decode_symbol2(input_stream<READ_COL_TYPE, in_buff_len>& s, queue<write_queue_ele>& mq, /*const dynamic_huffman* const huff_tree_ptr, unsigned buff_idx,*/
    const int16_t* const s_len, const int16_t* const lensym_ptr, const int16_t* const s_distcnt, const int16_t* const s_distsym) {


    while(1){
              
        //uint16_t sym = decode<READ_COL_TYPE, in_buff_len>(s,  s_len,  huff_tree_ptr[buff_idx].lensym);
        uint16_t sym = decode<READ_COL_TYPE, in_buff_len>(s,  s_len,  lensym_ptr);


        //parse 5 bits
        //not compressed, literal
        if(sym <= 255) {
            write_queue_ele qe;
            qe.type = 0;
            qe.data = (uint32_t) sym;
            //mq.enqueue(&qe);
            g_enqueue<write_queue_ele>(&qe, &mq);
                        
        }

        //end of block
        else if(sym == 256) {
            break;
        }

        //lenght, need to parse

        else{

           // uint16_t extra_bits = g_lext[sym - 257];

            uint32_t extra_len  = 0;
            if( g_lext[sym - 257] != 0){
               s.template fetch_n_bits<uint32_t>( g_lext[sym - 257], &extra_len);
            }

            uint16_t len = extra_len + g_lens[sym - 257];
            //distance, 5bits
            uint16_t sym_dist = decode<READ_COL_TYPE, in_buff_len>(s,  s_distcnt, s_distsym);    
            //uint16_t extra_bits_dist = g_dext[sym_dist];
            
            uint32_t extra_len_dist = 0;
            if(g_dext[sym_dist] != 0){
                s.template fetch_n_bits<uint32_t>(g_dext[sym_dist], &extra_len_dist);
            }


           //uint16_t dist = extra_len_dist + g_dists[sym_dist];

            write_queue_ele qe;
            qe.data = (len << 16) | (extra_len_dist + g_dists[sym_dist]);
            qe.type = 1;
            //mq.enqueue(&qe);
            g_enqueue<write_queue_ele>(&qe, &mq);

        }
    }

}


//if dynamic
//build tree
//
//decode length/distance pairs or literal
//
template <typename READ_COL_TYPE, size_t in_buff_len = 4>
//__forceinline__ 
__device__
void decoder_warp(input_stream<READ_COL_TYPE, in_buff_len>& s,  queue<write_queue_ele>& mq, uint32_t col_len, uint8_t* out, dynamic_huffman* huff_tree_ptr,
    slot_struct* d_slot_struct, const fix_huffman* const fixed_tree, int16_t* s_len, int16_t* s_distcnt, int16_t* s_distsym, int16_t* s_off) {

    
    unsigned sm_id;

    uint8_t slot = 0;
    if(threadIdx.x == 0){
       sm_id = get_smid();
       slot = find_slot(sm_id, d_slot_struct);
    }

    slot = __shfl_sync(FULL_MASK, slot, 0);
    sm_id = __shfl_sync(FULL_MASK, sm_id, 0);



    uint8_t blast;
    uint32_t btype;

    do{

    s.template fetch_n_bits<uint32_t>(19, &btype);
    btype >>= 16;
    blast =  (btype & 0x01);
    btype >>= 1;
    //fixed huffman
    if(btype == 1) {
        //decode_symbol<READ_COL_TYPE, in_buff_len>(s, mq, fixed_tree, 0);
        decode_symbol2<READ_COL_TYPE, in_buff_len> (s, mq,  fixed_tree -> lencnt, fixed_tree -> lensym, fixed_tree -> distcnt, fixed_tree -> distsym);
    }
    //dyamic huffman
    else{
       // uint32_t buff_idx =((sm_id * 32 + slot) * 32 + threadIdx.x);
        decode_dynamic<READ_COL_TYPE, in_buff_len>(s, huff_tree_ptr,  (uint32_t)((sm_id * 32 + slot) * 32 + threadIdx.x), s_len, s_distcnt, s_distsym, s_off);
       // decode_symbol<READ_COL_TYPE, in_buff_len>(s, mq, huff_tree_ptr, (uint32_t) ((sm_id * 32 + slot) * 32 + threadIdx.x));
        decode_symbol2<READ_COL_TYPE, in_buff_len>(s, mq, 
            s_len, huff_tree_ptr[((sm_id * 32 + slot) * 32 + threadIdx.x)].lensym, s_distcnt, s_distsym);

    }
  

    }while(blast != 1);

    __syncwarp(FULL_MASK);
    if(threadIdx.x == 0){

        release_slot(sm_id, slot, d_slot_struct);
    }

}


template <size_t WRITE_COL_LEN = 512, size_t CHUNK_SIZE = 8192>
//__forceinline__ 
__device__
void writer_warp_8div_warp2(queue<write_queue_ele>& mq, decompress_output<WRITE_COL_LEN, CHUNK_SIZE>& out, int division) {
    uint8_t div = 0;
    uint32_t MASK = 0;
    if(threadIdx.x < 4){
        MASK = MASK_8_1;
        div = 0;
    }
    else if(threadIdx.x < 4*2){
        MASK = MASK_8_2;
        div = 1;
    }
    else if(threadIdx.x < 4*3){
        MASK = MASK_8_3;
        div = 2;
    }
    else if(threadIdx.x < 4*4){
        MASK = MASK_8_4;
        div = 3;
    }
    else if(threadIdx.x < 4*5){
        MASK = MASK_8_5;
        div = 4;
    }
    else if(threadIdx.x < 4*6){
        MASK = MASK_8_6;
        div = 5;
    }
    else if(threadIdx.x < 4*7){
        MASK = MASK_8_7;
        div = 6;
    }
    else{
        MASK = MASK_8_8;
        div = 7;
    }

    //uint32_t total = CHUNK_SIZE / 32;
    bool done = false;
    while (!done) {

        bool deq = false;
        //uint64_t v = 0;
        write_queue_ele v;
        if(threadIdx.x % 2 == division)
            mq.attempt_dequeue(&v, &deq);

        uint32_t deq_mask = __ballot_sync(MASK, deq);
        uint8_t deq_count = __popc(deq_mask);


        for (int i = 0; i < deq_count; i++) {
            uint8_t f = __ffs(deq_mask);
            uint8_t t = __shfl_sync(MASK, v.type, f-1);
            uint32_t d = __shfl_sync(MASK, v.data, f-1);

            //pair
            if(t == 1){
                //uint16_t len = d >> 16;
                //uint16_t offset = (d) & 0x0000ffff;
                out.template col_memcpy_div<4>(f-1, (uint16_t)(d >> 16), (uint16_t)(d & 0x0000ffff), div, MASK);
            }
            //literal
            else{
                out.write_literal(f-1, (uint8_t)((d) & 0x00FF));
               
            }

            deq_mask >>= f;
            deq_mask <<= f;
        }
        //bool check =  (out.counter != out.total) && (threadIdx.x % 2 == division);
        // if(threadIdx.x % 2 != division)
        //     check = false;
        done = __ballot_sync(MASK, (out.counter != out.total) && (threadIdx.x % 2 == division)) == 0;


    } 

}



template <typename READ_COL_TYPE, typename COMP_COL_TYPE,  uint16_t in_queue_size = 4, size_t out_queue_size = 4, size_t local_queue_size = 4,  size_t WRITE_COL_LEN = 512, size_t CHUNK_SIZE = 8192>
__global__ void 
//__launch_bounds__ (96, 13)
//__launch_bounds__ (128, 12)
inflate(uint8_t* comp_ptr, const uint64_t* const col_len_ptr, const uint64_t* const blk_offset_ptr, uint8_t*out, dynamic_huffman* huff_tree_ptr,
 slot_struct* d_slot_struct, const fix_huffman* const fixed_tree) {
    __shared__ READ_COL_TYPE in_queue_[32][in_queue_size];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> h[32];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> t[32];

    __shared__ write_queue_ele out_queue_[32][out_queue_size];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> out_h[32];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> out_t[32];


    __shared__ READ_COL_TYPE local_queue[32][local_queue_size];

  

    //MAXDCODES is 30
    __shared__ int16_t s_distsym[32][MAXDCODES];
    __shared__ int16_t s_off[32][16];
    __shared__ int16_t s_lencnt[32][16];
    __shared__ int16_t s_distcnt[32][16];

    uint64_t col_len = (col_len_ptr[32 * blockIdx.x + threadIdx.x]);

    h[threadIdx.x] = 0;
    t[threadIdx.x] = 0;
    out_h[threadIdx.x] = 0;
    out_t[threadIdx.x] = 0;

    __syncthreads();

    if (threadIdx.y == 0) {
        queue<READ_COL_TYPE> in_queue(in_queue_[threadIdx.x], h + threadIdx.x , t + threadIdx.x, in_queue_size);

        //uint64_t blk_off = blk_offset_ptr[blockIdx.x];
        uint8_t* chunk_ptr = (comp_ptr +  blk_offset_ptr[blockIdx.x]);

        decompress_input<READ_COL_TYPE, COMP_COL_TYPE> d(chunk_ptr, col_len);
        reader_warp<READ_COL_TYPE, COMP_COL_TYPE>(d, in_queue);
    }

    else if (threadIdx.y == 1) {
    
        queue<READ_COL_TYPE> in_queue(in_queue_[threadIdx.x], h + threadIdx.x, t + threadIdx.x, in_queue_size);
        queue<write_queue_ele> out_queue(out_queue_[threadIdx.x], out_h + threadIdx.x, out_t + threadIdx.x, out_queue_size);
        input_stream<READ_COL_TYPE, local_queue_size> s(&in_queue, (uint32_t)col_len, local_queue[threadIdx.x]);
       
        decoder_warp<READ_COL_TYPE, local_queue_size>(s, out_queue, (uint32_t) col_len, out, huff_tree_ptr, d_slot_struct, fixed_tree, s_lencnt[threadIdx.x], s_distcnt[threadIdx.x], s_distsym[threadIdx.x], s_off[threadIdx.x]);
    }

    else {
        queue<write_queue_ele> out_queue(out_queue_[threadIdx.x], out_h + threadIdx.x, out_t + threadIdx.x, out_queue_size);
        decompress_output<WRITE_COL_LEN, CHUNK_SIZE> d((out + CHUNK_SIZE * blockIdx.x));
   //     writer_warp_8div<WRITE_COL_LEN, CHUNK_SIZE>(out_queue, d);
        writer_warp_8div_warp2<WRITE_COL_LEN, CHUNK_SIZE>(out_queue, d, threadIdx.y - 2);

    }

    __syncthreads();


}



namespace deflate {

template <typename READ_COL_TYPE>
 __host__ void decompress_gpu(const uint8_t* const in, uint8_t** out, const uint64_t in_n_bytes, uint64_t* out_n_bytes,
  uint64_t* col_len_f, const uint64_t col_n_bytes, uint64_t* blk_offset_f, const uint64_t blk_n_bytes) {

    uint64_t num_blk = ((uint64_t) blk_n_bytes / sizeof(uint64_t)) - 1;

    printf("num blk: %llu\n", num_blk);
    printf("READ_COL_TYPE size: %llu\n", (unsigned long) sizeof(READ_COL_TYPE));
    uint8_t* d_in;
    uint64_t* d_col_len;
    uint64_t* d_blk_offset;

    uint64_t* d_comp_histo;
    uint64_t* d_tree_histo;

    int num_sm = 108;
   
    cuda_err_chk(cudaMalloc(&d_in, in_n_bytes));
    cuda_err_chk(cudaMalloc(&d_col_len,col_n_bytes));
    cuda_err_chk(cudaMalloc(&d_blk_offset, blk_n_bytes));

    cuda_err_chk(cudaMemcpy(d_in, in, in_n_bytes, cudaMemcpyHostToDevice));
    cuda_err_chk(cudaMemcpy(d_col_len, col_len_f, col_n_bytes, cudaMemcpyHostToDevice));
    cuda_err_chk(cudaMemcpy(d_blk_offset, blk_offset_f, blk_n_bytes, cudaMemcpyHostToDevice));

    cuda_err_chk(cudaMalloc(&d_comp_histo, 256 * sizeof(d_comp_histo)));
    cuda_err_chk(cudaMemset(d_comp_histo, 0, 256));
    cuda_err_chk(cudaMalloc(&d_tree_histo, 2 * sizeof(d_comp_histo)));
    cuda_err_chk(cudaMemset(d_tree_histo, 0, 2));


    const size_t chunk_size = 8192 * 2;
    uint64_t out_bytes = chunk_size * num_blk;
    printf("out_bytes: %llu\n", out_bytes);

    uint8_t* d_out;
    *out_n_bytes = out_bytes;
    cuda_err_chk(cudaMalloc(&d_out, out_bytes));


    dynamic_huffman* d_tree;
    cuda_err_chk(cudaMalloc(&d_tree, sizeof(dynamic_huffman) * 32*num_sm*32));

    slot_struct* d_slot_struct;
    cuda_err_chk(cudaMalloc(&d_slot_struct, sizeof(slot_struct) * num_sm));
    cuda_err_chk(cudaMemset(d_slot_struct, 0, num_sm * sizeof(slot_struct)));



    uint16_t fix_lencnt[16] = {0,0,0,0,0,0,0,24,152,112,0,0,0,0,0,0};
    uint16_t fix_lensym[FIXLCODES] =
    { 256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,
        36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,
        69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,
        102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,
        127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,280,281,282,283,284,285,286,287,
        144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,
        169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,
        194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,
        219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,
        244,245,246,247,248,249,250,251,252,253,254,255};
     uint16_t fix_distcnt[MAXBITS + 1] = 
    {0,0,0,0,0,30,0,0,0,0,0,0,0,0,0,0};
    uint16_t fix_distsym[MAXDCODES] = 
    {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29};

    fix_huffman f_tree;
    fix_huffman* d_f_tree;

    cuda_err_chk(cudaMalloc(&d_f_tree, sizeof(fix_huffman)));

    //f_tree.lencnt = fix_lencnt;


    memcpy(f_tree.lencnt, fix_lencnt, sizeof(uint16_t)*16);
    memcpy(f_tree.lensym, fix_lensym, sizeof(uint16_t)*FIXLCODES);
    memcpy(f_tree.distcnt, fix_distcnt, sizeof(uint16_t)*(MAXBITS + 1));
    memcpy(f_tree.distsym, fix_distsym, sizeof(uint16_t)*MAXDCODES);

    cuda_err_chk(cudaMemcpy(d_f_tree, &f_tree, sizeof(fix_huffman), cudaMemcpyHostToDevice));

    printf("start inflation\n");

    dim3 blockD(32,4,1);
    dim3 gridD(num_blk,1,1);
    cudaDeviceSynchronize();

    std::chrono::high_resolution_clock::time_point kernel_start = std::chrono::high_resolution_clock::now();

    inflate<uint64_t, READ_COL_TYPE, 4 , 4, 2, 512, chunk_size> <<<gridD,blockD>>> (d_in, d_col_len, d_blk_offset, d_out, d_tree, d_slot_struct, d_f_tree);
    cuda_err_chk(cudaDeviceSynchronize());
    std::chrono::high_resolution_clock::time_point kernel_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total = std::chrono::duration_cast<std::chrono::duration<double>>(kernel_end - kernel_start);
    std::cout << "kernel time: " << total.count() << " secs\n";


    cudaError_t error = cudaGetLastError();
      if(error != cudaSuccess)
      {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
      }
    printf("inflation done\n");


    *out = new uint8_t[out_bytes];
    cuda_err_chk(cudaMemcpy((*out), d_out, out_bytes, cudaMemcpyDeviceToHost));

    uint64_t* h_histo = new uint64_t[256];
    uint64_t* h_tree_hist = new uint64_t[2];
    cuda_err_chk(cudaMemcpy(h_histo, d_comp_histo, 256 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    cuda_err_chk(cudaMemcpy(h_tree_hist, d_tree_histo, 2 * sizeof(uint64_t), cudaMemcpyDeviceToHost));


    //historgrams
    // printf("Compression pair histogram\n");
    // for(int i  = 0; i < 256; i++){
    //     printf("len: %i \t count: %llu\n", i, h_histo[i]);
    // }

    printf("huffman tree ratio\n");
    printf("fixed huffman: %llu \t dynamic huffman: %llu\n", h_tree_hist[0], h_tree_hist[1]);
    



    cuda_err_chk(cudaFree(d_out));
    cuda_err_chk(cudaFree(d_in));
    cuda_err_chk(cudaFree(d_col_len));
    cuda_err_chk(cudaFree(d_blk_offset));
 }


}

//#endif // __ZLIB_H__

