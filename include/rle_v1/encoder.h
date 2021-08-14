
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



struct  write_queue_ele{
    int8_t data;
    int header;
    bool done;
};



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
        //if(threadIdx.x == 0 && blockIdx.x == 0)      printf("deque start\n");

        const auto cur_head = head->load(simt::memory_order_relaxed);
        while (cur_head == tail->load(simt::memory_order_acquire))
            __nanosleep(50);

        *v = queue_[cur_head];

        const auto next_head = (cur_head + 1) % len;

        head->store(next_head, simt::memory_order_release);

    }


    
};





template <typename DECOMP_COL_TYPE>
struct compress_output {

    DECOMP_COL_TYPE* out_ptr;
    uint64_t offset;
    bool write_flag;

    __device__
    compress_output(uint8_t* ptr):
        out_ptr((DECOMP_COL_TYPE*)ptr) {
        offset = 0;
        write_flag = true;
    }


    __inline__ __device__
    uint64_t warp_write(DECOMP_COL_TYPE data) {

        uint32_t write_sync = __ballot_sync(FULL_MASK, write_flag);
        uint64_t idx = 0;
        if(write_flag){
            idx = offset + __popc(write_sync & (0xffffffff >> (32 - threadIdx.x)));
            out_ptr[idx] = data;
            offset += __popc(write_sync);
        }
        return idx;
    }

    __inline__ __device__
    void set_flag(){
        write_flag = true;
    }

    __device__
    uint64_t get_offset(){
        return offset;

    }


    __device__ 
    void update_header(uint64_t offset, uint64_t inword_offset, uint8_t head){
        uint8_t* byte_out = (uint8_t*) out_ptr;
        byte_out[offset * sizeof(DECOMP_COL_TYPE) + inword_offset] = head;

    }

};




template <typename READ_COL_TYPE, typename DATA_TYPE, size_t COMP_COL_LEN>
__device__
void compression_reader_warp(queue<DATA_TYPE>& rq, DATA_TYPE* input_data_ptr, uint64_t CHUNK_SIZE) {

    uint32_t num_iterations = (CHUNK_SIZE / 32 + COMP_COL_LEN - 1) / COMP_COL_LEN;
    uint32_t num_data_per_col = COMP_COL_LEN / sizeof(DATA_TYPE);
    uint32_t offset = threadIdx.x * num_data_per_col;

    for(int iter = 0; iter < num_iterations; iter++){
        for(int i = 0; i < num_data_per_col; i++){
            DATA_TYPE read_data = input_data_ptr[offset + i];
            rq.enqueue(&read_data);
        }
        offset += (num_data_per_col * 32);
    }
    return;
}


template <typename DATA_TYPE>
__device__ uint64_t get_varint_size(DATA_TYPE val){
    DATA_TYPE write_val = val;
    uint64_t out_len = 0;

    do {
        out_len++;
        write_val = write_val >> 7;
     } while (write_val != 0);

     return out_len;
}


template <typename DATA_TYPE>
__device__ void enqueue_varint(queue<write_queue_ele>& wq, DATA_TYPE val){
    DATA_TYPE write_val = val;
    int8_t write_byte = 0;
    do {
        write_byte = write_val & 0x7F;
    if ((write_val & (~0x7f)) != 0)
        write_byte = write_byte | 0x80;

        write_queue_ele write_data;
        write_data.done = false;
        write_data.header = 0;
        write_data.data = write_byte;

        wq.enqueue(&write_data);
        write_val = write_val >> 7;
     } while (write_val != 0);
}

__device__ void enqueue_byte(queue<write_queue_ele>& wq, uint8_t val){

    write_queue_ele write_data;
    write_data.done = false;
    write_data.header = 0;
    write_data.data = val;
    wq.enqueue(&write_data);
   
}

__device__ void enqueue_header_holder(queue<write_queue_ele>& wq,  int8_t val){
    write_queue_ele write_data;
    write_data.done = false;
    write_data.header = 1;
    write_data.data = val;
    wq.enqueue(&write_data);
}

__device__ void enqueue_header_setting(queue<write_queue_ele>& wq,  int8_t val){
    write_queue_ele write_data;
    if(threadIdx.x == 9 && blockIdx.x == 6) printf("set head: %x\n", val);
    write_data.done = false;
    write_data.header = 2;
    write_data.data = val;
    wq.enqueue(&write_data);
}



template <typename READ_COL_TYPE, typename DATA_TYPE, size_t COMP_COL_LEN, size_t in_buff_len = 4>
__device__
void compression_init_warp(queue<DATA_TYPE>& rq, uint64_t* col_len_ptr, uint64_t* blk_offset_ptr, uint64_t CHUNK_SIZE) {

    uint32_t read_bytes = sizeof(DATA_TYPE);

    uint64_t used_bytes = 0;
    uint64_t my_chunk_size = CHUNK_SIZE / 32;

    DATA_TYPE data_buffer[2];
    uint8_t data_buffer_head = 0;
    uint8_t data_buffer_count = 0;

    DATA_TYPE delta_first_val = 0;

    uint16_t delta_count = 0;
    int8_t cur_delta = 0;
    bool delta_flag = false;
    uint64_t lit_idx = 0;
    uint16_t lit_count = 0;
    DATA_TYPE prev_val = 0;
    
    uint64_t out_offset = threadIdx.x * COMP_COL_LEN;
    uint64_t out_len = 0;

    //first data
    DATA_TYPE read_data = 0;
    rq.dequeue(&read_data);
                            //if(threadIdx.x == 9 && blockIdx.x == 6) printf("readdata: %x\n", read_data);

    delta_count = 1;
    prev_val = read_data;

    data_buffer[data_buffer_head] = read_data;
    data_buffer_head = (data_buffer_head + 1) % 2;
    data_buffer_count++;
    used_bytes += read_bytes;

    rq.dequeue(&read_data);
                            //f(threadIdx.x == 9 && blockIdx.x == 6) printf("readdata: %x\n", read_data);


    //second data
    int64_t temp_diff = read_data - prev_val;

    if (temp_diff > 127 || temp_diff < -128) {
        delta_flag = false;
        lit_idx = out_offset;

        //enqueue_header_holder(wq, 1);
        out_len += 1;
        lit_count = 1;

        DATA_TYPE lit_val = data_buffer[0];
    
        out_len += get_varint_size<DATA_TYPE>( lit_val);

        data_buffer_count--;
    }

    else {
        delta_flag = true;
        cur_delta = (int8_t)temp_diff;
                        //if(threadIdx.x == 17 && blockIdx.x == 7 ) printf("cur delta: %x temp_diff %lx  \n",  cur_delta, temp_diff);

        delta_count++;
    }

    prev_val = read_data;
    data_buffer[data_buffer_head] = read_data;
    data_buffer_head = (data_buffer_head + 1) % 2;
    data_buffer_count++;
    used_bytes += read_bytes;




    while(used_bytes < my_chunk_size) {
       // if(threadIdx.x == 24 && blockIdx.x == 93064 ) printf("out len: %llu\n", out_len);

        if(lit_count == 127){
            //out_buffer_ptr[lit_idx] = static_cast<int8_t>(-lit_count);
            //enqueue_header_setting(wq, static_cast<int8_t>(-lit_count) );
            lit_count = 0;
        }

        rq.dequeue(&read_data);
                                //    if(threadIdx.x == 9 && blockIdx.x == 6) printf("readdata: %x\n", read_data);


        used_bytes += read_bytes;
        
        if(delta_count == 1){

            temp_diff = read_data - prev_val;
            if (temp_diff > 127 || temp_diff < -128) {
                delta_flag = false;
                if (lit_count == 0) {
                    lit_idx = out_offset;
                    //enqueue_header_holder(wq, 1);
                    out_len += 1;
                }

                int8_t data_buffer_tail = (data_buffer_head == 0) ? 1 : 0;
                DATA_TYPE lit_val = data_buffer[data_buffer_tail];
                out_len += get_varint_size<DATA_TYPE>( lit_val);

                lit_count++;
                data_buffer_count--;
            } 
            else {
                delta_flag = true;
                cur_delta = (int8_t)temp_diff;
                                                        //if(threadIdx.x == 17 && blockIdx.x == 7 ) printf("cur delta3: %x temp_diff %lx  \n",  cur_delta, temp_diff);

                delta_count++;
            }
            prev_val = read_data;
            data_buffer[data_buffer_head] = read_data;
            data_buffer_head = (data_buffer_head + 1) % 2;
            data_buffer_count++;

            continue;
        }
        
        //matched
        if(prev_val + cur_delta == read_data && delta_flag){
            delta_count++;
            if(delta_count == 3){
                                        //if(threadIdx.x == 17 && blockIdx.x == 7 ) printf("cur delta2: %x temp_diff %lx  \n",  cur_delta, temp_diff);

                delta_first_val = data_buffer[data_buffer_head];
            
                if(lit_count != 0){
                       // if(threadIdx.x == 9 && blockIdx.x == 6) printf("set head1:");

                   // enqueue_header_setting(wq, static_cast<int8_t>(-lit_count));
                    //out_buffer_ptr[lit_idx] = static_cast<int8_t>(-lit_count);
                    lit_count = 0;
                }
            }
            //max
            else if(delta_count == 130){
                int8_t write_byte = delta_count - 4;
                //enqueue_byte(wq, write_byte);
                write_byte = (uint8_t)cur_delta;

                //enqueue_byte(wq, write_byte);

                out_len += 2;
                out_len += get_varint_size( delta_first_val);

                delta_count = 1;
                data_buffer_count = 0;
                lit_count = 0;
            }

            data_buffer[data_buffer_head] = read_data;
            data_buffer_head = (data_buffer_head + 1) % 2;
            data_buffer_count = min(data_buffer_count + 1, 2);
            prev_val = read_data;
        }


        //not matched
        
        else{
            if(delta_count >= 3){

                int8_t write_byte = delta_count - 3;

                //enqueue_byte(wq, write_byte);

                write_byte = (uint8_t)cur_delta;

               // enqueue_byte(wq, write_byte);
            
                out_len += 2;
                out_len += get_varint_size<DATA_TYPE>( delta_first_val);
                
                //if(threadIdx.x == 17 && blockIdx.x == 7 ) printf("write_byte: %x cur delta:%x  fv: %x\n", delta_count - 3, cur_delta, delta_first_val);

                delta_count = 1;
                data_buffer_count = 0;
                lit_count = 0;
                data_buffer[data_buffer_head] = read_data;
                data_buffer_head = (data_buffer_head + 1) % 2;
                data_buffer_count = min(data_buffer_count + 1, 2);
                prev_val = read_data;
            }

            else{
                if(lit_count == 0){
                    lit_idx = out_offset;
                    //write_byte_op(out_buffer_ptr, &out_bytes, (uint8_t) 3,  &out_offset, s_col_len, &col_counter, COMP_WRITE_BYTES);
                   // enqueue_header_holder(wq, 1);
                    out_len ++;
                }
                lit_count++;
                DATA_TYPE lit_val = data_buffer[data_buffer_head];

               // write_varint(wq, lit_val);
                out_len += get_varint_size<DATA_TYPE>( lit_val);

                if(lit_count == 127){
                    //out_buffer_ptr[lit_idx] = static_cast<int8_t>(-lit_count);
                                          //  if(threadIdx.x == 9 && blockIdx.x == 6) printf("set head2:");

                   // enqueue_header_setting(wq, static_cast<int8_t>(-lit_count));

                    lit_count = 0;
                }


                int64_t temp_diff = read_data - prev_val;
                if (temp_diff > 127 || temp_diff < -128) {

                    if (lit_count == 0) {
                      lit_idx = out_offset;
                      //enqueue_header_holder(wq,1);
                      out_len++;
                      // write_byte_op(out_buffer_ptr, &out_bytes, (uint8_t) 3,  &out_offset,
                      //                          s_col_len, &col_counter, COMP_WRITE_BYTES);
                    }

                    delta_flag = false;
                    data_buffer_count = 0;
                    int8_t data_buffer_tail = (data_buffer_head == 0) ? 1 : 0;
                    DATA_TYPE lit_val = data_buffer[data_buffer_tail];
                    //enqueue_varint(wq, lit_val);
                                    out_len += get_varint_size<DATA_TYPE>( lit_val);

                    lit_count++;
                    delta_count = 1;
                }
                else {
                    data_buffer_count = 1;
                    delta_flag = true;
                    cur_delta = (int8_t) temp_diff;
                    delta_count = 2;
                }

                prev_val = read_data;
                data_buffer[data_buffer_head] = read_data;
                data_buffer_head = (data_buffer_head + 1) % 2;
                data_buffer_count = min(data_buffer_count + 1, 2);
            }
            
        }

    }

    if (delta_count >= 3 && delta_flag) {
        int8_t write_byte = delta_count - 3;
        //enqueue_byte(wq, write_byte);

        write_byte = (uint8_t)cur_delta;
        //enqueue_byte(wq, write_byte);
        out_len += 2;
        out_len += get_varint_size<DATA_TYPE>( delta_first_val);
        //if(threadIdx.x == 24 && blockIdx.x == 93064 ) printf("out len1: %llu\n", out_len);


    }

    else {
      // update lit count
     

      if(data_buffer_count == 1){
         if(lit_count == 127)
            lit_count = 0;

        if(lit_count == 0)
            out_len++;


        int8_t data_buffer_tail = (data_buffer_head == 0) ? 1 : 0;
        DATA_TYPE lit_val = data_buffer[data_buffer_tail];
       // enqueue_varint(wq, lit_val);
                                            out_len += get_varint_size<DATA_TYPE>( lit_val);
                                  //  if(threadIdx.x == 24 && blockIdx.x == 93064 ) printf("out len4: %llu\n", out_len);


        lit_count++;
      }

      if (data_buffer_count == 2) {
     
        if (lit_count == 127)
          lit_count = 0;

        if (lit_count == 0)
          out_len++;
        DATA_TYPE lit_val = data_buffer[data_buffer_head];
       // enqueue_varint(wq, lit_val);
                                            out_len += get_varint_size<DATA_TYPE>( lit_val);


        lit_count++;

        if (lit_count == 127)
          lit_count = 0;

        if (lit_count == 0)
            out_len++;

        data_buffer_head = (data_buffer_head + 1) % 2;
        lit_val = data_buffer[data_buffer_head];
        //enqueue_varint(wq, lit_val);
                                    out_len += get_varint_size<DATA_TYPE>( lit_val);
                                  //  if(threadIdx.x == 24 && blockIdx.x == 93064 ) printf("out len6: %llu\n", out_len);

        lit_count++;
      }
                            //  if(threadIdx.x == 9 && blockIdx.x == 6) printf("set head3:");

      //enqueue_header_setting(wq, (-lit_count));
       // if(threadIdx.x == 24 && blockIdx.x == 93064 ) printf("out len2: %llu\n", out_len);

    }


    col_len_ptr[threadIdx.x + 32 * blockIdx.x] = out_len;


    out_len = ((out_len + 4 - 1) / 4) * 4;


    __syncwarp();
    for (int offset = 16; offset > 0; offset /= 2)
        out_len += __shfl_down_sync(FULL_MASK, out_len, offset);


    __syncwarp();
    if(threadIdx.x == 0){
        out_len = ((out_len + 128 - 1) / 128) * 128;

        if(blockIdx.x == 643) printf("col len: %llu\n", out_len);
        
        blk_offset_ptr[blockIdx.x + 1] = out_len;        
    }

}



template <typename READ_COL_TYPE, typename DATA_TYPE, size_t COMP_COL_LEN, size_t in_buff_len = 4>
__device__
void compression_warp(queue<DATA_TYPE>& rq, queue<write_queue_ele>& wq, uint64_t* col_len_ptr, uint64_t* blk_offset_ptr, uint64_t CHUNK_SIZE) {

    uint32_t read_bytes = sizeof(DATA_TYPE);

    uint64_t used_bytes = 0;
    uint64_t my_chunk_size = CHUNK_SIZE / 32;

    DATA_TYPE data_buffer[2];
    uint8_t data_buffer_head = 0;
    uint8_t data_buffer_count = 0;

    DATA_TYPE delta_first_val = 0;

    uint16_t delta_count = 0;
    int8_t cur_delta = 0;
    bool delta_flag = false;
    uint64_t lit_idx = 0;
    uint16_t lit_count = 0;
    DATA_TYPE prev_val = 0;
    
    uint64_t out_offset = threadIdx.x * COMP_COL_LEN;

    //first data
    DATA_TYPE read_data = 0;
    rq.dequeue(&read_data);
                            //if(threadIdx.x == 9 && blockIdx.x == 6) printf("readdata: %x\n", read_data);

    delta_count = 1;
    prev_val = read_data;

    data_buffer[data_buffer_head] = read_data;
    data_buffer_head = (data_buffer_head + 1) % 2;
    data_buffer_count++;
    used_bytes += read_bytes;

    rq.dequeue(&read_data);
                            //f(threadIdx.x == 9 && blockIdx.x == 6) printf("readdata: %x\n", read_data);


    //second data
    int64_t temp_diff = read_data - prev_val;

    if (temp_diff > 127 || temp_diff < -128) {
        delta_flag = false;
        lit_idx = out_offset;

        enqueue_header_holder(wq, 1);

        lit_count = 1;

        DATA_TYPE lit_val = data_buffer[0];
        enqueue_varint<DATA_TYPE>(wq, lit_val);

        data_buffer_count--;
    }

    else {
        delta_flag = true;
        cur_delta = (int8_t)temp_diff;
                        //if(threadIdx.x == 17 && blockIdx.x == 7 ) printf("cur delta: %x temp_diff %lx  \n",  cur_delta, temp_diff);

        delta_count++;
    }

    prev_val = read_data;
    data_buffer[data_buffer_head] = read_data;
    data_buffer_head = (data_buffer_head + 1) % 2;
    data_buffer_count++;
    used_bytes += read_bytes;




    while(used_bytes < my_chunk_size) {

        if(lit_count == 127){
            //out_buffer_ptr[lit_idx] = static_cast<int8_t>(-lit_count);
            enqueue_header_setting(wq, static_cast<int8_t>(-lit_count) );
            lit_count = 0;
        }

        rq.dequeue(&read_data);
                                //    if(threadIdx.x == 9 && blockIdx.x == 6) printf("readdata: %x\n", read_data);


        used_bytes += read_bytes;
        
        if(delta_count == 1){

            temp_diff = read_data - prev_val;
            if (temp_diff > 127 || temp_diff < -128) {
                delta_flag = false;
                if (lit_count == 0) {
                    lit_idx = out_offset;
                    enqueue_header_holder(wq, 1);
                }

                int8_t data_buffer_tail = (data_buffer_head == 0) ? 1 : 0;
                DATA_TYPE lit_val = data_buffer[data_buffer_tail];
                enqueue_varint<DATA_TYPE>(wq, lit_val);

                lit_count++;
                data_buffer_count--;
            } 
            else {
                delta_flag = true;
                cur_delta = (int8_t)temp_diff;
                                                        //if(threadIdx.x == 17 && blockIdx.x == 7 ) printf("cur delta3: %x temp_diff %lx  \n",  cur_delta, temp_diff);

                delta_count++;
            }
            prev_val = read_data;
            data_buffer[data_buffer_head] = read_data;
            data_buffer_head = (data_buffer_head + 1) % 2;
            data_buffer_count++;

            continue;
        }
        
        //matched
        if(prev_val + cur_delta == read_data && delta_flag){
            delta_count++;
            if(delta_count == 3){
                                        //if(threadIdx.x == 17 && blockIdx.x == 7 ) printf("cur delta2: %x temp_diff %lx  \n",  cur_delta, temp_diff);

                delta_first_val = data_buffer[data_buffer_head];
            
                if(lit_count != 0){
                       // if(threadIdx.x == 9 && blockIdx.x == 6) printf("set head1:");

                    enqueue_header_setting(wq, static_cast<int8_t>(-lit_count));
                    //out_buffer_ptr[lit_idx] = static_cast<int8_t>(-lit_count);
                    lit_count = 0;
                }
            }
            //max
            else if(delta_count == 130){
                int8_t write_byte = delta_count - 4;
                enqueue_byte(wq, write_byte);
                write_byte = (uint8_t)cur_delta;

                enqueue_byte(wq, write_byte);
                enqueue_varint<DATA_TYPE>(wq, delta_first_val);

                delta_count = 1;
                data_buffer_count = 0;
                lit_count = 0;
            }

            data_buffer[data_buffer_head] = read_data;
            data_buffer_head = (data_buffer_head + 1) % 2;
            data_buffer_count = min(data_buffer_count + 1, 2);
            prev_val = read_data;
        }


        //not matched
        
        else{
            if(delta_count >= 3){

                int8_t write_byte = delta_count - 3;

                enqueue_byte(wq, write_byte);

                write_byte = (uint8_t)cur_delta;

                enqueue_byte(wq, write_byte);
                enqueue_varint<DATA_TYPE>(wq, delta_first_val);
                
                //if(threadIdx.x == 17 && blockIdx.x == 7 ) printf("write_byte: %x cur delta:%x  fv: %x\n", delta_count - 3, cur_delta, delta_first_val);

                delta_count = 1;
                data_buffer_count = 0;
                lit_count = 0;
                data_buffer[data_buffer_head] = read_data;
                data_buffer_head = (data_buffer_head + 1) % 2;
                data_buffer_count = min(data_buffer_count + 1, 2);
                prev_val = read_data;
            }

            else{
                if(lit_count == 0){
                    lit_idx = out_offset;
                    //write_byte_op(out_buffer_ptr, &out_bytes, (uint8_t) 3,  &out_offset, s_col_len, &col_counter, COMP_WRITE_BYTES);
                    enqueue_header_holder(wq, 1);
                }
                lit_count++;
                DATA_TYPE lit_val = data_buffer[data_buffer_head];

               // write_varint(wq, lit_val);
                enqueue_varint<DATA_TYPE>(wq, lit_val);

                if(lit_count == 127){
                    //out_buffer_ptr[lit_idx] = static_cast<int8_t>(-lit_count);
                                          //  if(threadIdx.x == 9 && blockIdx.x == 6) printf("set head2:");

                    enqueue_header_setting(wq, static_cast<int8_t>(-lit_count));

                    lit_count = 0;
                }


                int64_t temp_diff = read_data - prev_val;
                if (temp_diff > 127 || temp_diff < -128) {

                    if (lit_count == 0) {
                      lit_idx = out_offset;
                      enqueue_header_holder(wq,1);
                      // write_byte_op(out_buffer_ptr, &out_bytes, (uint8_t) 3,  &out_offset,
                      //                          s_col_len, &col_counter, COMP_WRITE_BYTES);
                    }

                    delta_flag = false;
                    data_buffer_count = 0;
                    int8_t data_buffer_tail = (data_buffer_head == 0) ? 1 : 0;
                    DATA_TYPE lit_val = data_buffer[data_buffer_tail];
                    enqueue_varint<DATA_TYPE>(wq, lit_val);

                    lit_count++;
                    delta_count = 1;
                }
                else {
                    data_buffer_count = 1;
                    delta_flag = true;
                    cur_delta = (int8_t) temp_diff;
                    delta_count = 2;
                }

                prev_val = read_data;
                data_buffer[data_buffer_head] = read_data;
                data_buffer_head = (data_buffer_head + 1) % 2;
                data_buffer_count = min(data_buffer_count + 1, 2);
            }
            
        }

    }

    if (delta_count >= 3 && delta_flag) {
        int8_t write_byte = delta_count - 3;
        enqueue_byte(wq, write_byte);

        write_byte = (uint8_t)cur_delta;
        enqueue_byte(wq, write_byte);
        enqueue_varint<DATA_TYPE>(wq, delta_first_val);
    }

    else {
      // update lit count
 

      if(data_buffer_count == 1){
        if(lit_count == 127) {
            enqueue_header_setting(wq, static_cast<int8_t>(-lit_count));
            lit_count = 0;
        }

        if(lit_count == 0){
            enqueue_header_holder(wq, 1);
        }

        int8_t data_buffer_tail = (data_buffer_head == 0) ? 1 : 0;
        DATA_TYPE lit_val = data_buffer[data_buffer_tail];
        enqueue_varint(wq, lit_val);
        lit_count++;
      }

      if (data_buffer_count == 2) {
        if(lit_count == 127) {
            enqueue_header_setting(wq, (-lit_count));
            lit_count = 0;
        }

        if(lit_count == 0){
            enqueue_header_holder(wq, 1);
        }


        DATA_TYPE lit_val = data_buffer[data_buffer_head];
        enqueue_varint<DATA_TYPE>(wq, lit_val);

        lit_count++;

        if(lit_count == 127) {
            enqueue_header_setting(wq, static_cast<int8_t>(-lit_count));
            lit_count = 0;
        }

        if(lit_count == 0){
            enqueue_header_holder(wq, 1);
        }


        data_buffer_head = (data_buffer_head + 1) % 2;
        lit_val = data_buffer[data_buffer_head];
        enqueue_varint<DATA_TYPE>(wq, lit_val);

        lit_count++;
      }
                            //  if(threadIdx.x == 9 && blockIdx.x == 6) printf("set head3:");

      enqueue_header_setting(wq, static_cast<int8_t>(-lit_count));
    }


    write_queue_ele done_data;
    done_data.data = 0;

    done_data.done = true;
    wq.enqueue(&done_data);

}


template <typename DECOMP_COL_TYPE, typename DATA_TYPE>
__device__
void compression_writer_warp(queue<write_queue_ele>& wq, compress_output< DECOMP_COL_TYPE>& out, uint64_t CHUNK_SIZE) {

    if(threadIdx.x == 0 && blockIdx.x == 0) printf("size of decomp type %i\n", sizeof(DECOMP_COL_TYPE));
    uint32_t done = 0;
    bool cur_done = false;
    DECOMP_COL_TYPE out_data = 0;
    uint64_t header_offset = 0;
    uint64_t offset_in_word = 0;
    int counter = 0;
    bool header_flag = false;
    int header_word_idx = 0;
    uint64_t iter = 0;

     while(!done){
        out_data = 0;
        for(int i = 0; i < sizeof(DECOMP_COL_TYPE);){

            write_queue_ele deq_data;

            if(!cur_done){
                wq.dequeue(&deq_data);
            



                if(deq_data.done == true){
                    out.set_flag();
                    cur_done = true;
                }


                else{
                    if(threadIdx.x == 0 && blockIdx.x == 91613 ) printf("header: %i\n", deq_data.header);

                    if(deq_data.header == 1){
                        //header_offset =  out.get_offset();
                        offset_in_word = i;
                        header_flag = true;
                        header_word_idx = iter;
                        deq_data.data = 0;
                    }

                    if(deq_data.header == 2){
                       // if(header_word_idx == 0) printf("tid: %i bid: %i head: %x\n",threadIdx.x, blockIdx.x,  deq_data.data);

                        if(iter == header_word_idx){
                            if(threadIdx.x == 0 && blockIdx.x == 91613 ) printf("header update: %i\n", (deq_data.data & 0x00FF));

                            DECOMP_COL_TYPE cur_data = (deq_data.data & 0x00FF);
                            out_data = out_data | (cur_data << (offset_in_word * 8));

                        }
                        else{
                            if(threadIdx.x == 0 && blockIdx.x == 91613 ) printf("header update: %i\n", (deq_data.data & 0x00FF));

                            out.update_header(header_offset, offset_in_word, (uint8_t) (deq_data.data & 0x00FF));
                            header_flag = false;
                        }
                        continue;
                    }
                    
                    // if(threadIdx.x == 0 && blockIdx.x == 0){
                    //     printf("data: %x\n", deq_data.data);
                    // }

           
                    DECOMP_COL_TYPE cur_data = (deq_data.data & 0x00FF);
                    
                   // if(threadIdx.x == 17 && blockIdx.x == 7 ) printf("data: %x\n", cur_data);
                    
                    counter++;
                    out_data = out_data | (cur_data << (i * 8));
                                       // if(threadIdx.x == 0 && blockIdx.x == 0 && counter < 20) printf("out i:%i offset: %i data: %x\n", i, (i * 8), out_data);
                    
                }
            }
            i++;
        }

        uint64_t cur_offset = out.warp_write(out_data);
        if(header_flag && header_word_idx == iter) header_offset = cur_offset;
        iter++;

        done = __ballot_sync(FULL_MASK, !cur_done) == 0;
    }


}



template <typename READ_COL_TYPE, typename DATA_TYPE, typename OUT_COL_TYPE, uint16_t in_queue_size = 4, size_t COMP_COL_LEN>
__global__ void 
setup_deflate(uint8_t* input_ptr,  uint64_t*  col_len_ptr, uint64_t*  blk_offset_ptr, uint64_t CHUNK_SIZE){

    static __shared__ DATA_TYPE in_queue_[32][in_queue_size];
    static __shared__ simt::atomic<uint8_t,simt::thread_scope_block> h[32];
    static __shared__ simt::atomic<uint8_t,simt::thread_scope_block> t[32];

    h[threadIdx.x] = 0;
    t[threadIdx.x] = 0;

    __syncthreads();



    if (threadIdx.y == 0) {
        queue<DATA_TYPE> in_queue(in_queue_[threadIdx.x], h + threadIdx.x , t + threadIdx.x, in_queue_size);
        uint8_t* chunk_ptr = (input_ptr +  (CHUNK_SIZE * blockIdx.x));
        compression_reader_warp<READ_COL_TYPE, DATA_TYPE, COMP_COL_LEN > (in_queue, (DATA_TYPE*) chunk_ptr, CHUNK_SIZE);
    }

    else if (threadIdx.y == 1) {
        queue<DATA_TYPE> in_queue(in_queue_[threadIdx.x], h + threadIdx.x, t + threadIdx.x, in_queue_size);
        compression_init_warp<READ_COL_TYPE, DATA_TYPE, COMP_COL_LEN, in_queue_size >(in_queue, col_len_ptr, blk_offset_ptr, CHUNK_SIZE);
    }


}


template <typename READ_COL_TYPE, typename DATA_TYPE, typename OUT_COL_TYPE, uint16_t queue_size = 4, size_t COMP_COL_LEN>
__global__ void 
deflate(uint8_t* input_ptr,  uint8_t* out,  uint64_t*  col_len_ptr, uint64_t*  blk_offset_ptr, uint64_t CHUNK_SIZE){

    static __shared__ DATA_TYPE in_queue_[32][queue_size];
    static __shared__ simt::atomic<uint8_t,simt::thread_scope_block> h[32];
    static __shared__ simt::atomic<uint8_t,simt::thread_scope_block> t[32];

    static __shared__ write_queue_ele out_queue_[32][queue_size];
    static __shared__ simt::atomic<uint8_t,simt::thread_scope_block> out_h[32];
    static __shared__ simt::atomic<uint8_t,simt::thread_scope_block> out_t[32];

    h[threadIdx.x] = 0;
    t[threadIdx.x] = 0;
    out_h[threadIdx.x] = 0;
    out_t[threadIdx.x] = 0;

    __syncthreads();

    if (threadIdx.y == 0) {
        queue<DATA_TYPE> in_queue(in_queue_[threadIdx.x], h + threadIdx.x , t + threadIdx.x, queue_size);
        uint8_t* chunk_ptr = (input_ptr +  (CHUNK_SIZE * blockIdx.x));
        compression_reader_warp<READ_COL_TYPE, DATA_TYPE, COMP_COL_LEN > (in_queue, (DATA_TYPE*) chunk_ptr, CHUNK_SIZE);
    }

    else if (threadIdx.y == 1) {
        queue<DATA_TYPE> in_queue(in_queue_[threadIdx.x], h + threadIdx.x, t + threadIdx.x, queue_size);
        queue<write_queue_ele> out_queue(out_queue_[threadIdx.x], out_h + threadIdx.x, out_t + threadIdx.x, queue_size);
        compression_warp<READ_COL_TYPE, DATA_TYPE, COMP_COL_LEN, queue_size >(in_queue, out_queue, col_len_ptr, blk_offset_ptr, CHUNK_SIZE);
    }
        else{
        queue<write_queue_ele> out_queue(out_queue_[threadIdx.x], out_h + threadIdx.x, out_t + threadIdx.x, queue_size);
        //compress_output<OUT_COL_TYPE> d((out + CHUNK_SIZE * blockIdx.x));
        compress_output<OUT_COL_TYPE> d((out + blk_offset_ptr[blockIdx.x]));
    
        compression_writer_warp<OUT_COL_TYPE, DATA_TYPE>(out_queue, d, CHUNK_SIZE);
    }

    __syncthreads();
}



__global__ void reduction_scan(uint64_t *blk_offset, uint64_t n) {
  blk_offset[0] = 0;
  for (int i = 1; i <= n; i++) {
    blk_offset[i] += blk_offset[i - 1];
        if(i-1 == 643)
            printf("blk offset: %llu\n", blk_offset[i-1]);

  }
}





