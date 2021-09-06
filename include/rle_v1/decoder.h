
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


template<typename T>
struct  decomp_write_queue_ele{
    T data;
    int header;
    bool done;
};



template <typename COMP_COL_TYPE >
struct decompress_input {


    //uint64_t col_width = sizeof(READ_COL_TYPE);
    uint16_t row_offset;
    uint16_t len;
    uint16_t read_bytes;
    //uint32_t t_read_mask;
   // READ_COL_TYPE* pointer;

    COMP_COL_TYPE* pointer;


    //READ_COL_TYPE in_buff[IN_BUFF_LEN];

    __device__
    decompress_input(const uint8_t* ptr, const uint16_t l) :
        pointer((COMP_COL_TYPE*) ptr), len(l) {
        //uint8_t tid = threadIdx.x;
      //  t_read_mask = (0xffffffff >> (32 - threadIdx.x));
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
            *v = pointer[row_offset + __popc(read_sync & (0xffffffff >> (32 - threadIdx.x)))];
            row_offset += __popc(read_sync);
           // if(threadIdx.x == 0 && blockIdx.x == 0) printf("row offset: %llu\n", (unsigned long long) row_offset);
            read_bytes += sizeof(COMP_COL_TYPE);
            read_count = sizeof(COMP_COL_TYPE);
        }

        __syncwarp(alivemask);

        return read_count;
    }



};



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
    input_stream(queue<READ_COL_TYPE>* q_, uint32_t eb, READ_COL_TYPE* shared_b) {
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
             q->dequeue(b.b + count);
            //g_dequeue<READ_COL_TYPE>(b.b + count, q);

        }
    }





    template<typename T>
    __device__
    void get_n_bits(const uint32_t n, T* out) {

        *out = (T) 0;

        ((uint32_t*) out)[0] = __funnelshift_rc(b.u[(uint_head)], b.u[(uint_head+1)%bu_size], uint_bit_offset);
        //uint32_t temp_out = __funnelshift_rc(b.u[(uint_head)], b.u[(uint_head+1)%bu_size], uint_bit_offset);


        if (32 > n) {
            ((uint32_t*) out)[0] <<= (32 - n);
            ((uint32_t*) out)[0] >>= (32 - n);
            // temp_out <<= (32 - n);
            // temp_out >>= (32 - n);
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
            //g_dequeue<READ_COL_TYPE>(b.b + ((head+count) % buff_len), q);
            q -> dequeue(b.b+((head+count) % buff_len) );
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
            q->dequeue(b.b + ((head+count) % buff_len));
            //g_dequeue<READ_COL_TYPE>(b.b + ((head+count) % buff_len), q);

            count++;
            uint_count += sizeof(READ_COL_TYPE)/sizeof(uint32_t);
            read_bytes += sizeof(READ_COL_TYPE);
        }
        *out = (T) 0;

         ((uint32_t*) out)[0] = __funnelshift_rc(b.u[(uint_head)], b.u[(uint_head+1)%bu_size], uint_bit_offset);
        //uint32_t temp_out = __funnelshift_rc(b.u[(uint_head)], b.u[(uint_head+1)%bu_size], uint_bit_offset);


        if (32 > n) {
            ((uint32_t*) out)[0] <<= (32 - n);
            ((uint32_t*) out)[0] >>= (32 - n);
            // temp_out <<= (32 - n);
            // temp_out >>= (32 - n);
        }



    }


};


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
void reader_warp(decompress_input< COMP_COL_TYPE>& in, queue<COMP_COL_TYPE>& rq) {
    while (true) {
        COMP_COL_TYPE v;
        int8_t rc = in.comp_read_data(FULL_MASK, &v);
        //int8_t rc = comp_read_data2(FULL_MASK, &v, in);

        if (rc == -1)
            break;
        else if (rc > 0){

            rq.enqueue(&(v));
            //comp_enqueue<COMP_COL_TYPE, READ_COL_TYPE>(&v, &rq);
        }
    }
}

template <typename COMP_COL_TYPE, typename DATA_TYPE, size_t in_buff_len = 4>
//__forceinline__ 
__device__
void decoder_warp(input_stream<COMP_COL_TYPE, in_buff_len>& s, queue<decomp_write_queue_ele<DATA_TYPE>>& mq,  decompress_output<DATA_TYPE>& out, uint64_t CHUNK_SIZE, DATA_TYPE* out_buf, int COMP_COL_LEN) {

    int test_idx = 40;

    uint32_t input_data_out_size = 0;
    uint64_t num_iterations = (CHUNK_SIZE / 32)  ;


    uint64_t words_in_line = COMP_COL_LEN / sizeof(DATA_TYPE);
    uint64_t out_offset = words_in_line * threadIdx.x;
    uint64_t c= 0;


    while (input_data_out_size < num_iterations) {
    
      //need to read a header
      int32_t temp_byte = 0;
      s.template fetch_n_bits<int32_t>(8, &temp_byte);

      int8_t head_byte = (int8_t)(temp_byte & 0x00FF);


      //literals
      if(head_byte < 0){
        uint64_t remaining = static_cast<uint64_t>(-head_byte);
            //if(threadIdx.x == test_idx && blockIdx.x == 1 ) printf("num_iterations: %llu remaining: %x\n", num_iterations, remaining);

        for(uint64_t i = 0; i < remaining; ++i){

            DATA_TYPE value = 0;
            int64_t offset = 0;

            
            //read var-int value
            bool read_next = true;
            while(read_next){
                temp_byte = 0;
                s.template fetch_n_bits<int32_t>(8, &temp_byte);
                int8_t in_data = 0;
                in_data = (int8_t) (in_data | (temp_byte & 0x00FF));

                if(in_data >= 0){
                    value |= (static_cast<DATA_TYPE>(in_data) << offset);
                    read_next = false;
                }
                else{
                    value |= ((static_cast<DATA_TYPE>(in_data) & 0x7f) << offset); 
                    offset += 7;
                }
            }

            decomp_write_queue_ele<DATA_TYPE> qe;
            qe.data = value;

            out_buf[out_offset] = value;

            c++;
            if(c == words_in_line){
                out_offset += words_in_line * 31;
                c=0;
            }
       
                out_offset++;
            



           // mq.enqueue(&qe);
           //(threadIdx.x == 0 && blockIdx.x == 0 && input_data_out_size <= 20) printf("out: %c\n", value);

            input_data_out_size+=sizeof(DATA_TYPE);

        }

      }
      //compresssed data
      else{
            uint64_t remaining = static_cast<uint64_t>(head_byte);

            temp_byte = 0;
            s.template fetch_n_bits<int32_t>(8, &temp_byte);
            int8_t delta = (int8_t) (temp_byte & 0x00FF);

            DATA_TYPE value = 0;
            int64_t offset = 0;

            
            int32_t in_data;

            while(1){
                temp_byte = 0;
                s.template fetch_n_bits<int32_t>(8, &temp_byte);

                int8_t in_data =  (int8_t) (temp_byte & 0x00FF);

                if(in_data >= 0){
                                   // if(threadIdx.x == test_idx && blockIdx.x == 0) printf("data4: %x\n", in_data);

                    value |= (static_cast<DATA_TYPE>(in_data) << offset);
                    break;
                }
                else{
                                  // if(threadIdx.x == test_idx && blockIdx.x == 0) printf("data5: %x\n", in_data);

                    value |= ((static_cast<DATA_TYPE>(in_data) & 0x7f) << offset); 
                    offset += 7;
                }
            }

            //decoding the compresssed stream
            for(uint64_t i = 0; i < remaining + 3; ++i){
                int64_t out_ele = value + static_cast<int64_t>(i) * delta;
                //write out_ele 

                //temp_write_word = temp_write_word | (static_cast<READ_T>(out_ele) << (temp_word_count * sizeof(INPUT_T) * 8));

                decomp_write_queue_ele<DATA_TYPE> qe;
                qe.data = static_cast<DATA_TYPE>(out_ele); 


                out_buf[out_offset] =  static_cast<DATA_TYPE>(out_ele);
                c++;
                 if(c == words_in_line){
                    out_offset += words_in_line * 31;
                    c=0;
                }
                    
                        out_offset++;
                                 
                //mq.enqueue(&qe);

                 input_data_out_size+=sizeof(DATA_TYPE);

            }


        }

    }


    //printf("yeah done!! bid:%i tid: %i\n", blockIdx.x, threadIdx.x);



}


template <typename COMP_COL_TYPE, typename DATA_TYPE, typename OUT_COL_TYPE, uint16_t queue_size = 4>
__global__ void 
//__launch_bounds__ (96, 13)
//__launch_bounds__ (128, 11)
inflate(uint8_t* comp_ptr, uint8_t* out, const uint64_t* const col_len_ptr, const uint64_t* const blk_offset_ptr, uint64_t CHUNK_SIZE, int COMP_COL_LEN) {

    __shared__ COMP_COL_TYPE in_queue_[32][queue_size];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> h[32];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> t[32];

    __shared__ COMP_COL_TYPE local_queue[32][2];

    uint64_t col_len = (col_len_ptr[32 * (blockIdx.x) + threadIdx.x]);

    h[threadIdx.x] = 0;
    t[threadIdx.x] = 0;
    out_h[threadIdx.x] = 0;
    out_t[threadIdx.x] = 0;

    //if(blockIdx.x == 1 && threadIdx.x == 0) printf("blk offset: %llu\n",blk_offset_ptr[blockIdx.x ] );

    __syncthreads();

    if (threadIdx.y == 0) {
        queue<COMP_COL_TYPE> in_queue(in_queue_[threadIdx.x], h + threadIdx.x , t + threadIdx.x, queue_size);
        uint8_t* chunk_ptr = (comp_ptr +  blk_offset_ptr[blockIdx.x ]);
        decompress_input< COMP_COL_TYPE> d(chunk_ptr, col_len);
        reader_warp<COMP_COL_TYPE>(d, in_queue);
    }

    else if (threadIdx.y == 1) {
    
        queue<COMP_COL_TYPE> in_queue(in_queue_[threadIdx.x], h + threadIdx.x, t + threadIdx.x, queue_size);
        queue<decomp_write_queue_ele<DATA_TYPE>> out_queue(out_queue_[threadIdx.x], out_h + threadIdx.x, out_t + threadIdx.x, queue_size);
        input_stream<COMP_COL_TYPE, 2> s(&in_queue, (uint32_t)col_len, local_queue[threadIdx.x]);
        decompress_output<DATA_TYPE> d((out + CHUNK_SIZE * (blockIdx.x )));
        decoder_warp<COMP_COL_TYPE, DATA_TYPE, 2>(s, out_queue, d, CHUNK_SIZE, (DATA_TYPE*)(out + CHUNK_SIZE * blockIdx.x), COMP_COL_LEN);

    }

    __syncthreads();
    //if(threadIdx.x == 0) printf("bid: %i done\n", blockIdx.x);
    // else{
    //     queue<write_queue_ele<DATA_TYPE>> out_queue(out_queue_[threadIdx.x], out_h + threadIdx.x, out_t + threadIdx.x, queue_size);
    //     decompress_output<OUT_COL_TYPE> d((out + CHUNK_SIZE * blockIdx.x));
    //     writer_warp<OUT_COL_TYPE, DATA_TYPE, CHUNK_SIZE>(out_queue, d);
    // }
}









