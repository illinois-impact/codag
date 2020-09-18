
#include <algorithm>
#include <cassert>
#include <common.h>
#include <cub/cub.cuh>
#include <fstream>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <simt/atomic>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

constexpr uint16_t THRDS_SM_() { return (2048); }
constexpr uint16_t BLK_SIZE_() { return (32); }
constexpr uint16_t BLKS_SM_() { return (THRDS_SM_() / BLK_SIZE_()); }
constexpr uint64_t GRID_SIZE_() { return (1024); }
constexpr uint64_t NUM_CHUNKS_() { return (GRID_SIZE_() * BLK_SIZE_()); }
constexpr uint64_t CHUNK_SIZE_() { return (2 * 1024); }
constexpr uint64_t INPUT_BUFFER_SIZE() { return (8); }
constexpr uint64_t CHUNK_SIZE_4() { return (128); }

constexpr uint64_t HEADER_SIZE_() { return (1); }
constexpr uint32_t OVERHEAD_PER_CHUNK_(uint32_t d) {
  return (ceil<uint32_t>(d, (HEADER_SIZE_() * 8)) + 1);
}
constexpr uint32_t HIST_SIZE_() { return 2048; }
constexpr uint32_t LOOKAHEAD_SIZE_() { return 512; }
constexpr uint32_t REF_SIZE_() { return 16; }
constexpr uint32_t REF_SIZE_BYTES_() { return REF_SIZE_() / 8; }
constexpr uint32_t OFFSET_SIZE_() {
  return (bitsNeeded((uint32_t)HIST_SIZE_()));
}
constexpr uint32_t LENGTH_SIZE_() { return (REF_SIZE_() - OFFSET_SIZE_()); }
constexpr uint32_t LENGTH_MASK_(uint32_t d) {
  return ((d > 0) ? 1 | (LENGTH_MASK_(d - 1)) << 1 : 0);
}
constexpr uint32_t MIN_MATCH_LENGTH_() {
  return (ceil<uint32_t>((OFFSET_SIZE_() + LENGTH_SIZE_()), 8) + 1);
}
constexpr uint32_t MAX_MATCH_LENGTH_() {
  return (pow<uint32_t, uint32_t>(2, LENGTH_SIZE_()) + MIN_MATCH_LENGTH_() - 1);
}
constexpr uint8_t DEFAULT_CHAR_() { return ' '; }
constexpr uint32_t HEAD_INTS_() { return 7; }
constexpr uint32_t READ_UNITS_() { return 4; }
constexpr uint32_t LOOKAHEAD_UNITS_() {
  return LOOKAHEAD_SIZE_() / READ_UNITS_();
}
constexpr uint64_t WARP_ID_(uint64_t t) { return t / 32; }
constexpr uint32_t LOOKAHEAD_SIZE_4_BYTES_() {
  return LOOKAHEAD_SIZE_() / sizeof(uint32_t);
}
constexpr uint32_t HIST_SIZE_4_BYTES_() {
  return HIST_SIZE_() / sizeof(uint32_t);
}

constexpr uint32_t CHUNK_SIZE_4_BYTES_MASK_() {
  return LENGTH_MASK_(bitsNeeded(CHUNK_SIZE_4()));
}

#define BLKS_SM BLKS_SM_()
#define THRDS_SM THRDS_SM_()
#define BLK_SIZE BLK_SIZE_()
#define GRID_SIZE GRID_SIZE_()
#define NUM_CHUNKS NUM_CHUNKS_()
#define CHUNK_SIZE CHUNK_SIZE_()
#define HEADER_SIZE HEADER_SIZE_()
#define OVERHEAD_PER_CHUNK(d) OVERHEAD_PER_CHUNK_(d)
#define HIST_SIZE HIST_SIZE_()
#define LOOKAHEAD_SIZE LOOKAHEAD_SIZE_()
#define OFFSET_SIZE OFFSET_SIZE_()
#define LENGTH_SIZE LENGTH_SIZE_()
#define LENGTH_MASK(d) LENGTH_MASK_(d)
#define MIN_MATCH_LENGTH MIN_MATCH_LENGTH_()
#define MAX_MATCH_LENGTH MAX_MATCH_LENGTH_()
#define DEFAULT_CHAR DEFAULT_CHAR_()
#define HEAD_INTS HEAD_INTS_()
#define READ_UNITS READ_UNITS_()
#define LOOKAHEAD_UNITS LOOKAHEAD_UNITS_()
#define WARP_ID(t) WARP_ID_(t)

#define INPUT_BUFFER_SIZE INPUT_BUFFER_SIZE()

#define CHUNK_SIZE_4_BYTES_MASK CHUNK_SIZE_4_BYTES_MASK_()

#define char_len 4

#define READING_WARP_SIZE 16
#define WRITING_WARP_SIZE 16

#define DATA_BUFFER_SIZE 64
#define NUM_THREADS 32

#define COMP_WRITE_BYTES 4

#define DECOMP_READING_WARP_SIZE 16

namespace brle_trans {



template <typename INPUT_T, typename READ_T>
__global__ void
rlev1_decompress_func(const int8_t *const in, INPUT_T* out, const
  uint64_t n_chunks, uint64_t* col_len, uint8_t *col_map, uint64_t *blk_offset) {

  __shared__ simt::atomic<uint8_t,  simt::thread_scope_block> in_head[32];
  __shared__ simt::atomic<uint8_t,  simt::thread_scope_block> in_tail[32];
      
  __shared__ READ_T input_buffer[READING_WARP_SIZE][32];

  __shared__ simt::atomic<INPUT_T, simt::thread_scope_block> out_head[32];
  __shared__ simt::atomic<INPUT_T, simt::thread_scope_block> out_tail[32];
  __shared__ INPUT_T out_buffer[WRITING_WARP_SIZE][32];


  __shared__  int t_done;

  if(threadIdx.x == 0) 
    t_done = 0;

  uint32_t which = threadIdx.y;
  int tid = threadIdx.x;
  int chunk_idx = blockIdx.x;
  uint64_t in_start_idx = blk_offset[chunk_idx];

  if(tid < 32 && which  == 0){
          in_head[tid] = 0;
          in_tail[tid] = 0;
          out_head[tid] = 0;
          out_tail[tid] = 0;
  }

   __syncthreads();



  if(which == 0){
    READ_T* in_READ = (READ_T *)(&(in[in_start_idx]));

    uint64_t mychunk_size = col_len[BLK_SIZE * chunk_idx + tid];
    uint64_t used_bytes = 0;
    uint32_t in_read_off = 0;



    while(used_bytes < mychunk_size){
      unsigned mask = __activemask(); 
      int res =  __popc(mask);
      __syncwarp(mask);

      READ_T temp_store = in_READ[in_read_off + tid];

      in_read_off += res;

      r_decomp_read:
          const auto cur_tail = in_tail[tid].load(simt::memory_order_relaxed);
          const auto next_tail = (cur_tail  + 1) % READING_WARP_SIZE;

          if (next_tail != in_head[tid].load(simt::memory_order_acquire)) {
    
            input_buffer[cur_tail][tid] = temp_store;

            if(blockIdx.x == 0 && tid == 15){
              printf("temp store: %x\n", temp_store);
            }

            in_tail[tid].store(next_tail, simt::memory_order_release);
            used_bytes += sizeof(READ_T);
          }
          else {
            __nanosleep(100);
            goto r_decomp_read;
          }
      __syncwarp(mask);
    }
     // if(blockIdx.x == 1 && threadIdx.x == 8){
     //           printf("read done\n" );
     //        }

  // printf("read bid:%i tid: %i done \n", blockIdx.x, tid);
              
  }

  else if(which == 1){
    
    uint64_t mychunk_size = CHUNK_SIZE / BLK_SIZE;
    uint64_t used_bytes = 0;
        uint64_t used_iterations = 0;


    int8_t data_buffer[DATA_BUFFER_SIZE];
    uint8_t data_buffer_head = 0;
    uint8_t data_buffer_tail = 0;
    uint8_t data_buffer_count = 0;

    uint8_t read_byte = sizeof(READ_T);
    uint8_t input_byte = sizeof(INPUT_T);

    bool header_read = true;
    bool read_val_flag = false;

    bool literal_flag = true;
    bool comp_flag = true;

    int8_t head_byte = 0;
    uint64_t remaining = 0;

    INPUT_T value = 0;
    //check
    uint64_t write_iterations = (CHUNK_SIZE / BLK_SIZE) / input_byte;

    uint8_t delta = 0;
    uint64_t out_start_idx = chunk_idx * CHUNK_SIZE;
    uint64_t output_off = 0;
    uint64_t col_idx = col_map[BLK_SIZE * chunk_idx + tid];

    // if(tid == 1){
    //   printf("bid: %i start\n", blockIdx.x);
    // }

    while (used_iterations < write_iterations) {
     
      if(data_buffer_count == 0){
      r_decomp_compute_read:
        const auto cur_head = in_head[tid].load(simt::memory_order_relaxed);
        if (cur_head == in_tail[tid].load(simt::memory_order_acquire)) {
          __nanosleep(100);
          goto r_decomp_compute_read;
        }


      READ_T read_data = input_buffer[cur_head][tid];
      const auto next_head = (cur_head + 1) % READING_WARP_SIZE;
      in_head[tid].store(next_head, simt::memory_order_release);

      READ_T temp_read_data = read_data;

          // if(blockIdx.x == 0 && threadIdx.x == 31){
          //    printf("start read data: %x  header_read: %i  read_val_flag: %i  comp_flag: %i\n",  read_data, header_read, read_val_flag, comp_flag);
          // }


      for(uint8_t i = 0; i < read_byte; i++){
        data_buffer[data_buffer_tail] = ((INPUT_T*)(&temp_read_data))[i];
        data_buffer_tail = (data_buffer_tail + 1) % DATA_BUFFER_SIZE;
      }

      data_buffer_count += read_byte;
    }
         //need to read a header
      if(header_read){
          head_byte = data_buffer[data_buffer_head];
          data_buffer_count --;
          data_buffer_head = (data_buffer_head + 1) % DATA_BUFFER_SIZE;
          header_read = false;


          //literals
          if(head_byte < 0){

            remaining = static_cast<uint64_t>(-head_byte);
            if(blockIdx.x == 0 && threadIdx.x == 2)
            // printf("remain: %llu\n", remaining);
           
            literal_flag = true;
            comp_flag = false;
          }
          //compresssed data
          else{
            remaining = static_cast<uint64_t>(head_byte);
            literal_flag = false;
            comp_flag = true;

            //may have to check the count

            if(data_buffer_count == 0){

              r_decomp_compute_read2:
                const auto cur_head = in_head[tid].load(simt::memory_order_relaxed);
                if (cur_head == in_tail[tid].load(simt::memory_order_acquire)) {
                  __nanosleep(100);
                  goto r_decomp_compute_read2;
                }

                  READ_T new_read_data = input_buffer[cur_head][tid];
                  const auto next_head = (cur_head + 1) % READING_WARP_SIZE;
                  in_head[tid].store(next_head, simt::memory_order_release);

                  READ_T temp_read_data = new_read_data;

                    //     if(blockIdx.x == 0 && threadIdx.x == ){
                    //    printf("delta read data: %x  header_read: %i  read_val_flag: %i  comp_flag: %i\n",  new_read_data, header_read, read_val_flag, comp_flag);
                    // }


                  
                    for(uint8_t i = 0; i < read_byte; i++){
                      data_buffer[data_buffer_tail] = ((INPUT_T*)(&temp_read_data))[i];
                      data_buffer_tail = (data_buffer_tail + 1) % DATA_BUFFER_SIZE;
                    }

                   data_buffer_count += read_byte;
            }


            delta = data_buffer[data_buffer_head];
            data_buffer_head = (data_buffer_head + 1) % DATA_BUFFER_SIZE;
            data_buffer_count--;

            read_val_flag = true;
          }
      }

     // if(blockIdx.x == 0 && threadIdx.x == 2)
//      printf("header done\n");

      if(read_val_flag){
        value = 0;
        int64_t offset = 0;

        while(read_val_flag){

          if(data_buffer_count == 0){
            r_value_read:
              const auto cur_head = in_head[tid].load(simt::memory_order_relaxed); 
              if (cur_head == in_tail[tid].load(simt::memory_order_acquire)) {
                __nanosleep(100);
                goto r_value_read;
              }


               READ_T read_data = input_buffer[cur_head][tid];
                const auto next_head = (cur_head + 1) % READING_WARP_SIZE;
                in_head[tid].store(next_head, simt::memory_order_release);

                    // if(blockIdx.x == 0 && threadIdx.x == 2){
                    //    printf("rv read data: %x  header_read: %i  read_val_flag: %i  comp_flag: %i\n",  read_data, header_read, read_val_flag, comp_flag);
                    // }


                READ_T temp_read_data = read_data;
               
          for(uint8_t i = 0; i < read_byte; i++){
            data_buffer[data_buffer_tail] = ((INPUT_T*)(&temp_read_data))[i];
            data_buffer_tail = (data_buffer_tail + 1) % DATA_BUFFER_SIZE;
          }

            data_buffer_count += read_byte;
          }


          int8_t in_data = data_buffer[data_buffer_head];
          data_buffer_head = (data_buffer_head + 1) % DATA_BUFFER_SIZE;
          data_buffer_count--;

          if(in_data >= 0){
            value |= (static_cast<INPUT_T>(in_data) << offset);
            read_val_flag = false;
          }
          else{
            value |= ((static_cast<INPUT_T>(in_data) & 0x7f) << offset); 
            offset += 7;
            }

          }
      }

    //  if(blockIdx.x == 0 && threadIdx.x == 2)
    //  printf("read val done\n");

      uint64_t count = remaining;
      


      if(comp_flag){


   //   if(blockIdx.x == 0 && threadIdx.x == 2)
    //  printf("remaining: %llu\n", remaining);


        for(uint64_t i = 0; i < remaining + 3; ++i){
          //push the element into the queue

     // if(blockIdx.x == 0 && threadIdx.x == 2)
     // printf("i: %llu\n", i);

          int64_t out_ele = value + static_cast<int64_t>(i) * delta;

          // r_compute_write_1:
          //   const auto cur_tail = out_tail[tid].load(simt::memory_order_relaxed); 
          //   const auto next_tail = (cur_tail + 1) % WRITING_WARP_SIZE;

          //   if (next_tail != out_head[tid].load(simt::memory_order_acquire)) {

          //     out_buffer[cur_tail][tid] = static_cast<INPUT_T>(out_ele);

          //     out_tail[tid].store(next_tail, simt::memory_order_release);
          //   }
          //   else {
          //     __nanosleep(100);
          //     goto r_compute_write_1;
          //   }

          out[out_start_idx + output_off + col_idx*sizeof(READ_T)] =  static_cast<INPUT_T>(out_ele);
          // if(blockIdx.x == 420 && threadIdx.x == 8)
          //  printf("out comp: %x\n",  static_cast<INPUT_T>(out_ele));
            output_off++;

           if(output_off % sizeof(READ_T) == 0){
            output_off += sizeof(READ_T) * 31;
           }
       

            used_iterations += 1;
        }
          header_read = true;
      }

      else{
        for(uint64_t i = 0; i < remaining; ++i){

          value = 0;
          int64_t offset = 0;
          read_val_flag = true;

          while(read_val_flag){

            if(data_buffer_count == 0){

              r_value_read2:
                const auto cur_head = in_head[tid].load(simt::memory_order_relaxed); 
                if (cur_head == in_tail[tid].load(simt::memory_order_acquire)) {
                    __nanosleep(100);
                    goto r_value_read2;
                }


              READ_T read_data = input_buffer[cur_head][tid];
              const auto next_head = (cur_head + 1) % READING_WARP_SIZE;
              in_head[tid].store(next_head, simt::memory_order_release);
             
              READ_T temp_read_data = read_data;
             
      for(uint8_t i = 0; i < read_byte; i++){
        data_buffer[data_buffer_tail] = ((INPUT_T*)(&temp_read_data))[i];
        data_buffer_tail = (data_buffer_tail + 1) % DATA_BUFFER_SIZE;
      }
              data_buffer_count += read_byte;

            }


            int8_t in_data = data_buffer[data_buffer_head];
            data_buffer_head = (data_buffer_head + 1) % DATA_BUFFER_SIZE;
            data_buffer_count--;

       
            if(in_data >= 0){
          
              value |= (static_cast<INPUT_T>(in_data) << offset);
              read_val_flag = false;
            }
            else{
                         

              value |= ((static_cast<INPUT_T>(in_data) & 0x7f) <<offset); 
              offset += 7;
            }
          }

          // r_compute_write_2:
          //   const auto cur_tail = out_tail[tid].load(simt::memory_order_relaxed); 
          //   const auto next_tail = (cur_tail+ 1) % WRITING_WARP_SIZE;

          //   if (next_tail != out_head[tid].load(simt::memory_order_acquire)) {

          //     out_buffer[cur_tail][tid] = static_cast<INPUT_T>(value);

          //     out_tail[tid].store(next_tail, simt::memory_order_release);
          //                     if(blockIdx.x == 0 && threadIdx.x == 2)
          //       printf("out: %x\n",  static_cast<INPUT_T>(value));
          //   }
          //   else {
          //     __nanosleep(100);
          //     goto r_compute_write_2;
          //   }

           out[out_start_idx + output_off + col_idx*sizeof(READ_T)] =  static_cast<INPUT_T>(value);
            //  if(blockIdx.x == 420 && threadIdx.x == 8)
            // printf("out lit: %x\n",  static_cast<INPUT_T>(value));
            output_off++;

           if(output_off % sizeof(READ_T) == 0){
            output_off += sizeof(READ_T) * 31;
           }
          
            used_iterations += 1;

        }

        header_read = true;
        read_val_flag = false;
      }

    }

        //if((blockIdx.x == 689))
    //     printf("bid:%i tid: %i done \n", blockIdx.x, tid);
              
        // atomicAdd(&t_done, 1);

        // if(t_done == 32 && blockIdx.x != 0){
        //   printf("blockIdx: %i done \n", blockIdx.x);
        // }


    
  }

  else if(which == 2){

    // if(blockIdx.x == 0 && threadIdx.x == 2)
    //       printf("start write \n" );
              

    uint64_t writes_phases = (CHUNK_SIZE / BLK_SIZE);
    int tid = threadIdx.x;
    uint32_t out_off = 0;
    uint64_t out_start_idx = chunk_idx * CHUNK_SIZE;
    uint64_t col_idx = col_map[BLK_SIZE * chunk_idx + tid];


    for(uint64_t i = 0; i < writes_phases; i++){
      r_decomp_write:
        const auto cur_head = out_head[tid].load(simt::memory_order_relaxed);
        if (cur_head == out_tail[tid].load(simt::memory_order_acquire)) {
          __nanosleep(100);
          goto r_decomp_write;
        }

        INPUT_T temp_out = out_buffer[cur_head][tid];

        const auto next_head = (cur_head + 1) % WRITING_WARP_SIZE;
        out_head[tid].store(next_head, simt::memory_order_release);

        __syncwarp();
        // writing based on col_idx
        out[out_start_idx + out_off + col_idx] = temp_out;
      
       // if(blockIdx.x == 0 && threadIdx.x == 2)
       //    printf("temp_out: %x\n",  temp_out);
              

        out_off += 32;
    }

  }




}




__device__ uint64_t roundUpTo(uint64_t input, uint64_t unit) {
  uint64_t val = ((input + unit - 1) / unit) * unit;
  return val;
}

__device__ void write_byte_op(int8_t *out_buffer, uint64_t *out_bytes_ptr,
                              uint8_t write_byte, uint64_t *out_offset_ptr,
                              uint64_t *col_len, uint8_t *col_counter_ptr) {

  out_buffer[*out_offset_ptr] = write_byte;

 if(threadIdx.x == 1 && blockIdx.x == 0){

         printf("write data: %x  off:%llu \n",  write_byte, *out_offset_ptr);
      }

  (*out_bytes_ptr) = (*out_bytes_ptr) + 1;

  // update the offset
  uint64_t out_offset = (*out_offset_ptr);

  if (( (*out_bytes_ptr) ) % COMP_WRITE_BYTES != 0) {
    *out_offset_ptr = out_offset + 1;
  }

  else {
    uint8_t col_counter = *col_counter_ptr;
    uint64_t col_start = blockIdx.x * BLK_SIZE;
    while ((*out_bytes_ptr) >
               roundUpTo(col_len[col_start + col_counter], COMP_WRITE_BYTES) &&
           col_counter > 0) {
      col_counter--;
    }
    out_offset += col_counter * COMP_WRITE_BYTES;

    *col_counter_ptr = col_counter;
    *out_offset_ptr = out_offset + 1;
  }
}

template <typename INPUT_T>
__device__ void write_varint_op(int8_t *out_buffer, uint64_t *out_bytes_ptr,
                                INPUT_T val, uint64_t *out_offset_ptr,
                                uint64_t *col_len, uint8_t *col_counter_ptr) {

  INPUT_T write_val = val;

  int8_t write_byte = 0;
  do {
    //uint8_t shift_literation = write_val / 128;
   // uint8_t temp = write_val >> (7 * shift_literation);

    write_byte = write_val & 0x7F;

    if ((write_val & (~0x7f)) != 0)
      write_byte = write_byte | 0x80;

    // write byte
    write_byte_op(out_buffer, out_bytes_ptr, write_byte, out_offset_ptr,
                  col_len, col_counter_ptr);

    write_val = write_val >> 7;
    //write_val = write_val & (0x00 << (7 * shift_literation));
  } while (write_val != 0);
}

template <typename INPUT_T, typename READ_T>
__global__ void
rlev1_compress_func_init(uint8_t *in, const uint64_t in_chunk_size,
                         const uint64_t n_chunks, uint64_t *col_len,
                         uint8_t *col_map, uint64_t *blk_offset) {

  __shared__ simt::atomic<uint8_t, simt::thread_scope_block> in_head[32];
  __shared__ simt::atomic<uint8_t, simt::thread_scope_block> in_tail[32];
  __shared__ READ_T in_buffer[READING_WARP_SIZE][32];

  int tid = threadIdx.x;
  int which = threadIdx.y;

  if (tid < 32 && which == 0) {
    in_head[tid] = 0;
    in_tail[tid] = 0;
  }

   __shared__ unsigned long long int block_len;
    if (threadIdx.x == 0) {
      block_len = 0;
      if (blockIdx.x == 0) {
        blk_offset[0] = 0;
      }
    }
  __syncthreads();

  // reading warp
  if (which == 0) {
    uint64_t in_start_idx = in_chunk_size * blockIdx.x;
    uint64_t mychunk_size = in_chunk_size / NUM_THREADS;
    READ_T *inTyped = (READ_T*)(&(in[in_start_idx]));

    uint64_t reading_used_bytes = 0;
    uint32_t in_off = 0;
    int tid = threadIdx.x;
    uint64_t ele_size = sizeof(READ_T);

    while (reading_used_bytes < mychunk_size) {
      unsigned mask = __activemask();
      int res = __popc(mask);
      __syncwarp(mask);

      READ_T temp_store = inTyped[in_off + tid];
      in_off += res;


    // if(blockIdx.x == 2 && threadIdx.x == 8 )
    //   printf("init data: %x\n", temp_store);

      // if(blockIdx.x == 13818 && col_idx == 3 )
      // printf("init data: %x\n", temp_store);


    r_read:
      const auto cur_tail = in_tail[tid].load(simt::memory_order_relaxed);
      const auto next_tail = (cur_tail + 1) % READING_WARP_SIZE;

      if (next_tail != in_head[tid].load(simt::memory_order_acquire)) {
        in_buffer[cur_tail][tid] = temp_store;
        in_tail[tid].store(next_tail, simt::memory_order_release);
        reading_used_bytes += ele_size;
      } else {
        __nanosleep(100);
        goto r_read;
      }
      __syncwarp(mask);
    }


    // comp_reading_warp_op<INPUT_T>(mychunk_size, inTyped, &(in_head[0]),
    // &(in_tail[0]),
    //                              (INPUT_T**)in_buffer);
  }

  // computational warp
  else if (which == 1) {

    uint64_t my_chunk_size = in_chunk_size / NUM_THREADS;
   
    int tid = threadIdx.x;
    int chunk_idx = blockIdx.x;

    INPUT_T data_buffer[2];
    uint8_t data_buffer_head = 0;
    uint8_t data_buffer_count = 0;

    uint64_t used_bytes = 0;
    uint64_t read_bytes = sizeof(INPUT_T);

    uint16_t delta_count = 0;

    int8_t cur_delta = 0;
    bool delta_flag = false;

    INPUT_T delta_first_val = 0;
    INPUT_T prev_val = 0;

    uint64_t lit_idx = 0;
    uint8_t lit_count = 0;

    uint64_t out_len = 0;

    const int buf_size = sizeof(READ_T)/sizeof(INPUT_T);

    INPUT_T input_buffer[buf_size];
    int input_buffer_count = 0;
    int input_buffer_head = 0;
  
    while (used_bytes < my_chunk_size) {

      if(input_buffer_count == 0){
        r_compute_read:
          const auto cur_head = in_head[tid].load(simt::memory_order_relaxed);
          if (cur_head == in_tail[tid].load(simt::memory_order_acquire)) {
            __nanosleep(100);
            goto r_compute_read;
          }

          READ_T in_read_data = in_buffer[cur_head][tid];
          const auto next_head = (cur_head + 1) % READING_WARP_SIZE;
          in_head[tid].store(next_head, simt::memory_order_release);


          for(int i = 0; i < buf_size; i++) {
            input_buffer[i] = ((INPUT_T*)(&in_read_data))[i];

            if(blockIdx.x == 0 && threadIdx.x == 1){
              printf("data: %x\n", input_buffer[i]);
            }

          }

          input_buffer_count = buf_size;
          input_buffer_head = 0;
      }
      INPUT_T read_data = input_buffer[input_buffer_head];
      input_buffer_head++;
      input_buffer_count--;

      // first element
      if (used_bytes == 0) {
        delta_count = 1;
        prev_val = read_data;

        data_buffer[data_buffer_head] = read_data;
        data_buffer_head = (data_buffer_head + 1) % 2;
        data_buffer_count++;
        used_bytes += read_bytes;
        continue;
      }

      // second element to fill the buffer
      else if (used_bytes == read_bytes) {

        int64_t temp_diff = read_data - prev_val;


        if (temp_diff > 127 || temp_diff < -128) {
          delta_flag = false;
          out_len++;
          lit_count = 1;

          INPUT_T lit_val = data_buffer[0];
          uint64_t val_bytes = (lit_val / 128) + 1;

          out_len += val_bytes;
          data_buffer_count--;


        }

        else {
          delta_flag = true;
          cur_delta = (int8_t)temp_diff;
          delta_count++;
        }

        prev_val = read_data;
        data_buffer[data_buffer_head] = read_data;
        data_buffer_head = (data_buffer_head + 1) % 2;
        data_buffer_count++;

        used_bytes += read_bytes;

        continue;
      }

      used_bytes += read_bytes;

      if (delta_count == 1) {
        int64_t temp_diff = read_data - prev_val;


        if (temp_diff > 127 || temp_diff < -128) {

          delta_flag = false;

          if (lit_count == 0) {
            out_len++;
            lit_count = 1;
        
          }

          int8_t data_buffer_tail = (data_buffer_head == 0) ? 1 : 0;

          INPUT_T lit_val = data_buffer[data_buffer_tail];
          uint64_t val_bytes = (lit_val / 128) + 1;

          out_len += val_bytes;
          lit_count++;


          data_buffer_count--;




        } else {
          delta_flag = true;
          cur_delta = (int8_t)temp_diff;
          delta_count++;
        }

        prev_val = read_data;
        data_buffer[data_buffer_head] = read_data;
        data_buffer_head = (data_buffer_head + 1) % 2;
        data_buffer_count++;
        continue;
      }

      if (prev_val + cur_delta == read_data && delta_flag) {

        delta_count++;
        if (delta_count == 3) {
          delta_first_val = data_buffer[data_buffer_head];
          if (lit_count != 0) {
            lit_count = 0;
          }
        }
        data_buffer[data_buffer_head] = read_data;
        data_buffer_head = (data_buffer_head + 1) % 2;
        data_buffer_count = min(data_buffer_count + 1, 2);
         prev_val = read_data;
      }

      else {

        if (delta_count >= 3) {



          // write count, del, val
          out_len += 2;
          uint64_t val_bytes = (delta_first_val / 128) + 1;
          out_len += val_bytes;


          // if(threadIdx.x == 0 && blockIdx.x == 0){
          //   printf("read delta_first_val1: %x  val_bytes:%llu,  out_len: %llu\n", delta_first_val, val_bytes, out_len);

          // }

          delta_count = 1;
          data_buffer_count = 0;
          lit_count = 0;

          data_buffer[data_buffer_head] = read_data;
          data_buffer_head = (data_buffer_head + 1) % 2;
          data_buffer_count = min(data_buffer_count + 1, 2);
          prev_val = read_data;

        } else {

          // first lit val
          if (lit_count == 0) {
            out_len++;
            // if(threadIdx.x == 0 && blockIdx.x == 0){
            //  printf("label2 lit_count0  out_len: %llu\n", out_len);
            // }
          }
          lit_count++;

          // write lit
          INPUT_T lit_val = data_buffer[data_buffer_head];
          uint64_t val_bytes = (lit_val / 128) + 1;
          out_len += val_bytes;

          // if(threadIdx.x == 0 && blockIdx.x == 0){
          //     printf("valbytes 2 lit_val:%x  out_len: %llu\n", lit_val, out_len);
          //   }

          int64_t temp_diff = read_data - prev_val;

          if (temp_diff > 127 || temp_diff < -128) {
            delta_flag = false;
            data_buffer_count = 0;

            int8_t data_buffer_tail = (data_buffer_head == 0) ? 1 : 0;
            INPUT_T lit_val = data_buffer[data_buffer_tail];
            uint64_t val_bytes = (lit_val / 128) + 1;

            out_len += val_bytes;
            //  if(threadIdx.x == 0 && blockIdx.x == 0) {
            //   printf("no\n");
            // }
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



    // write remaining elements
    if (delta_count >= 3 && delta_flag) {


 
      out_len += 2;
      int num_out_bytes = (delta_first_val / 128) + 1;
      out_len += num_out_bytes;


    }

    else {
      // update lit count
      if (lit_count == 0) {
        out_len++;
      }
    
      if(data_buffer_count == 1){
        int data_buffer_tail = (data_buffer_head == 0) ? 1 : 0;
        INPUT_T lit_val = data_buffer[data_buffer_tail];
        int num_out_bytes = (lit_val / 128) + 1;
        out_len += num_out_bytes;
        data_buffer_head = (data_buffer_head + 1) % 2;


      }

      if (data_buffer_count == 2) {

        INPUT_T lit_val = data_buffer[data_buffer_head];
        int num_out_bytes = (lit_val / 128) + 1;
        out_len += num_out_bytes;
        data_buffer_head = (data_buffer_head + 1) % 2;


        lit_val = data_buffer[data_buffer_head];
        num_out_bytes = (lit_val / 128) + 1;
        out_len += num_out_bytes;
      }

    }


    col_len[BLK_SIZE * chunk_idx + tid] = out_len;
    uint64_t out_len_round = roundUpTo(out_len, COMP_WRITE_BYTES);

    atomicAdd((unsigned long long int *)&block_len,
              (unsigned long long int)out_len_round);
    __syncthreads();
    if (threadIdx.x == 0) {
      // 128B alignment
      block_len = roundUpTo(block_len, 128);
      blk_offset[chunk_idx + 1] = (uint64_t)block_len;
    }

    //comp_computational_warp_init_op<INPUT_T>(in_head, in_tail,
    //                                         (INPUT_T **)in_buffer, col_len,
    //                                        col_map, blk_offset, mychunk_size);
  }
}

template <typename INPUT_T, typename READ_T>
__global__ void
rlev1_compress_func(uint8_t * in, int8_t *out_buffer,
                    // const uint64_t in_n_bytes, uint64_t *out_n_bytes,
                    const uint64_t in_chunk_size, const uint64_t n_chunks,
                    uint64_t *col_len, uint8_t *col_map, uint64_t *blk_offset) {

  __shared__ simt::atomic<uint8_t, simt::thread_scope_block> in_head[32];
  __shared__ simt::atomic<uint8_t, simt::thread_scope_block> in_tail[32];
  __shared__ READ_T in_buffer[READING_WARP_SIZE][32];


  int tid = threadIdx.x;
  int which = threadIdx.y;
  uint8_t col_idx;
  int chunk_idx = blockIdx.x;


  if (tid < 32 && which == 0) {
    in_head[tid] = 0;
    in_tail[tid] = 0;
  }

 for (int i = 0; i < 32; i++) {
      if (tid == col_map[BLK_SIZE * chunk_idx + i]) {
        col_idx = i;
      }
    }


  __syncthreads();
  // reading warp
  if (which == 0) {
    // uint64_t in_start_idx = in_chunk_size * blockIdx.x;
    // uint64_t mychunk_size = in_chunk_size / NUM_THREADS;
    // INPUT_T *inTyped = (&(in[in_start_idx]));
    // comp_reading_warp_op<INPUT_T>(mychunk_size, inTyped, in_head, in_tail,
    //                               in_buffer);

    uint64_t in_start_idx = in_chunk_size * blockIdx.x;
    // if(threadIdx.x == 0 && blockIdx.x == 1){
    //   printf("chunk: %llu\n",in_chunk_size);
    //   printf("isi: %llu\n", in_start_idx);
    // }
    
    uint64_t mychunk_size = in_chunk_size / NUM_THREADS;
    READ_T *inTyped = (READ_T*) (&(in[in_start_idx]));

    uint64_t reading_used_bytes = 0;
    uint32_t in_off = 0;
    int tid = threadIdx.x;
    uint64_t ele_size = sizeof(READ_T);

    while (reading_used_bytes < mychunk_size) {
      unsigned mask = __activemask();
      int res = __popc(mask);
      __syncwarp(mask);

      READ_T temp_store = inTyped[in_off + tid];
      in_off += res;

    r_read:
      const auto cur_tail = in_tail[tid].load(simt::memory_order_relaxed);
      const auto next_tail = (cur_tail + 1) % READING_WARP_SIZE;

      if (next_tail != in_head[tid].load(simt::memory_order_acquire)) {
        in_buffer[cur_tail][tid] = temp_store;
        in_tail[tid].store(next_tail, simt::memory_order_release);
        reading_used_bytes += ele_size;
      } else {
        __nanosleep(100);
        goto r_read;

      }
      __syncwarp(mask);
    }

  }

  // computational warp
  else if (which == 1) {
    // uint64_t mychunk_size = in_chunk_size / NUM_THREADS;
    // comp_computational_warp_op<INPUT_T>(in_head, in_tail, in_buffer, out,
    //                                     col_len, col_map, blk_offset,
    //                                     mychunk_size);
    uint64_t my_chunk_size = in_chunk_size / NUM_THREADS;

    int tid = threadIdx.x;

    INPUT_T data_buffer[2];
    uint8_t data_buffer_head = 0;
    uint8_t data_buffer_count = 0;

    uint64_t used_bytes = 0;
    uint64_t read_bytes = sizeof(INPUT_T);

    uint16_t delta_count = 0;

    int8_t cur_delta = 0;
    bool delta_flag = false;

    INPUT_T delta_first_val = 0;
    INPUT_T prev_val = 0;

    uint64_t lit_idx = 0;
    int8_t lit_count = 0;

    uint8_t col_counter = 31;
    uint64_t out_bytes = 0;
    uint64_t out_offset = col_idx * COMP_WRITE_BYTES;

    uint64_t out_start_idx = blk_offset[chunk_idx];

   
    //change
    int8_t *out_buffer_ptr =
        &(out_buffer[out_start_idx]);

    const int buf_size = sizeof(READ_T)/sizeof(INPUT_T);

    INPUT_T input_buffer[buf_size];
    int input_buffer_count = 0;
    int input_buffer_head = 0;
        if(col_idx == 0 && blockIdx.x == 689){
          printf("col idx,,,, tid: %i\n",tid);
        }

    while (used_bytes < my_chunk_size) {

 
      if(input_buffer_count == 0){
        r_compute_read:
          const auto cur_head = in_head[tid].load(simt::memory_order_relaxed);
          if (cur_head == in_tail[tid].load(simt::memory_order_acquire)) {
            __nanosleep(100);
            goto r_compute_read;
          }

          READ_T in_read_data = in_buffer[cur_head][tid];
          const auto next_head = (cur_head + 1) % READING_WARP_SIZE;
          in_head[tid].store(next_head, simt::memory_order_release);


          for(int i = 0; i < buf_size; i++) {
            input_buffer[i] = ((INPUT_T*)(&in_read_data))[i];
          }

          input_buffer_count = buf_size;
          input_buffer_head = 0;
      }
      INPUT_T read_data = input_buffer[input_buffer_head];
      input_buffer_head++;
      input_buffer_count--;


      // if(threadIdx.x == 28 && blockIdx.x == 420){
      //    printf("read data: %x lit_count:%i delta_count: %u delta_flag:%i cur_delta:%i \n",  read_data, lit_count, delta_count, delta_flag, cur_delta);
      // }
         if(col_idx == 0 && blockIdx.x == 689){
         printf("read data: %x lit_count:%i delta_count: %u delta_flag:%i cur_delta:%i \n",  read_data, lit_count, delta_count, delta_flag, cur_delta);
      }
      // first element
      if (used_bytes == 0) {
        delta_count = 1;
        prev_val = read_data;

        data_buffer[data_buffer_head] = read_data;
        data_buffer_head = (data_buffer_head + 1) % 2;
        data_buffer_count++;
        used_bytes += read_bytes;
        continue;
      }

      // second element to fill the buffer
      else if (used_bytes == read_bytes) {

        int64_t temp_diff = read_data - prev_val;

        if (temp_diff > 127 || temp_diff < -128) {
          delta_flag = false;
          lit_idx = out_offset;
          write_varint_op<INPUT_T>(out_buffer_ptr, &out_bytes, 1, &out_offset,
                                   col_len, &col_counter);
          lit_count = 1;

          INPUT_T lit_val = data_buffer[0];
          write_varint_op<INPUT_T>(out_buffer_ptr,  &out_bytes, lit_val,
                                   &out_offset, col_len, &col_counter);

          data_buffer_count--;
        }

        else {
          delta_flag = true;
          cur_delta = (int8_t)temp_diff;
          delta_count++;
        }

        prev_val = read_data;
        data_buffer[data_buffer_head] = read_data;
        data_buffer_head = (data_buffer_head + 1) % 2;
        data_buffer_count++;
        used_bytes += read_bytes;

        continue;
      }

      used_bytes += read_bytes;

      if (delta_count == 1) {
        int64_t temp_diff = read_data - prev_val;

        if (temp_diff > 127 || temp_diff < -128) {

          delta_flag = false;

          if (lit_count == 0) {
            lit_idx = out_offset;
            write_byte_op(out_buffer_ptr,  &out_bytes, (uint8_t)1, &out_offset,
                                     col_len, &col_counter);
           // lit_count = 1;
          }

          int8_t data_buffer_tail = (data_buffer_head == 0) ? 1 : 0;

          INPUT_T lit_val = data_buffer[data_buffer_tail];
          write_varint_op<INPUT_T>(out_buffer_ptr, &out_bytes, lit_val, 
                                   &out_offset, col_len, &col_counter);
          lit_count++;

          data_buffer_count--;

        } else {
          delta_flag = true;
          cur_delta = (int8_t)temp_diff;
          delta_count++;
        }

        prev_val = read_data;
        data_buffer[data_buffer_head] = read_data;
        data_buffer_head = (data_buffer_head + 1) % 2;
        data_buffer_count++;


        continue;
      }

      if (prev_val + cur_delta == read_data && delta_flag) {

        delta_count++;
        if (delta_count == 3) {
          delta_first_val = data_buffer[data_buffer_head];
          if (lit_count != 0) {
            // update lit counter

      

            out_buffer_ptr[lit_idx] = static_cast<int8_t>(-lit_count);
            lit_count = 0;
          }
        }
        data_buffer[data_buffer_head] = read_data;
        data_buffer_head = (data_buffer_head + 1) % 2;
        data_buffer_count = min(data_buffer_count + 1, 2);
        prev_val = read_data;

      }

      else {
     
        
        if (delta_count >= 3) {

          // write count, del, val
          int8_t write_byte = delta_count - 3;
          write_byte_op(out_buffer_ptr, &out_bytes, write_byte, &out_offset,
                        col_len, &col_counter);


          write_byte = (uint8_t)cur_delta;
          write_byte_op(out_buffer_ptr, &out_bytes, write_byte, &out_offset,
                        col_len, &col_counter);
          
   
          write_varint_op<INPUT_T>(out_buffer_ptr, &out_bytes, delta_first_val, 
                                   &out_offset, col_len, &col_counter);
   
          delta_count = 1;
          data_buffer_count = 0;
          lit_count = 0;

          data_buffer[data_buffer_head] = read_data;
          data_buffer_head = (data_buffer_head + 1) % 2;
          data_buffer_count = min(data_buffer_count + 1, 2);
          prev_val = read_data;

        } else {

          // first lit val
 
          if (lit_count == 0) {
            lit_idx = out_offset;
            write_byte_op(out_buffer_ptr, &out_bytes, (uint8_t) 3,  &out_offset,
                                     col_len, &col_counter);
          }
          lit_count++;

          // write lit
          INPUT_T lit_val = data_buffer[data_buffer_head];
          write_varint_op<INPUT_T>(out_buffer_ptr, &out_bytes, lit_val,
                                   &out_offset, col_len, &col_counter);

          int64_t temp_diff = read_data - prev_val;

          if (temp_diff > 127 || temp_diff < -128) {
            delta_flag = false;
            data_buffer_count = 0;

            int8_t data_buffer_tail = (data_buffer_head == 0) ? 1 : 0;
            INPUT_T lit_val = data_buffer[data_buffer_tail];
            write_varint_op<INPUT_T>(out_buffer_ptr,  &out_bytes, lit_val,
                                     &out_offset, col_len, &col_counter);
            

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
 

    // write remaining elements
    if (delta_count >= 3 && delta_flag) {

      int8_t write_byte = delta_count - 3;

      write_byte_op(out_buffer_ptr, &out_bytes, write_byte, &out_offset,  col_len,
                    &col_counter);

      write_byte = (uint8_t)cur_delta;
      write_byte_op(out_buffer_ptr, &out_bytes, write_byte, &out_offset, col_len,
                    &col_counter);

      write_varint_op<INPUT_T>(out_buffer_ptr,  &out_bytes, delta_first_val,
                               &out_offset, col_len, &col_counter);

    } else {


     

      // update lit count
      if (lit_count == 0) {
        lit_idx = out_offset;
        write_varint_op<INPUT_T>(out_buffer_ptr, &out_bytes, 1,  &out_offset,
                                 col_len, &col_counter);
      }

      if(data_buffer_count == 1){
        int8_t data_buffer_tail = (data_buffer_head == 0) ? 1 : 0;
            INPUT_T lit_val = data_buffer[data_buffer_tail];
          write_varint_op<INPUT_T>(out_buffer_ptr, &out_bytes, lit_val, &out_offset,
                                 col_len, &col_counter);
          lit_count++;
      }

      if (data_buffer_count == 2) {
        INPUT_T lit_val = data_buffer[data_buffer_head];
        write_varint_op<INPUT_T>(out_buffer_ptr, &out_bytes, lit_val, &out_offset,
                               col_len, &col_counter);
        lit_count++;

        data_buffer_head = (data_buffer_head + 1) % 2;
        lit_val = data_buffer[data_buffer_head];
        write_varint_op<INPUT_T>(out_buffer_ptr, &out_bytes, lit_val, &out_offset,
                                 col_len, &col_counter);
        lit_count++;
      }

      // if(tid== 25 && blockIdx.x == 13818)
      //   printf("l count: %x\n", lit_count);
        //   if(tid == 9 && blockIdx.x == 33){
        //   printf("lit_idx:%x lit count: %x\n", lit_idx, lit_count);
        // }


      out_buffer_ptr[lit_idx] = (-lit_count);
    }



  }
}

__global__ void ExampleKernel(uint64_t *col_len, uint8_t *col_map,
                              uint64_t *out) {
  // Specialize BlockRadixSort for a 1D block of 128 threads owning 4 integer
  // keys and values each
  typedef cub::BlockRadixSort<uint64_t, 32, 1, int> BlockRadixSort;
  __shared__ typename BlockRadixSort::TempStorage temp_storage;

  int tid = threadIdx.x;
  int bid = blockIdx.x;

  uint64_t thread_keys[1];
  int thread_values[1];
  thread_keys[0] = col_len[bid * BLK_SIZE + tid];
  thread_values[0] = tid;

  BlockRadixSort(temp_storage).Sort(thread_keys, thread_values);

  // col_len[bid*BLK_SIZE + tid] = thread_keys[0];
  col_map[bid * BLK_SIZE + BLK_SIZE - 1 - tid] = thread_values[0];
  out[bid * BLK_SIZE + BLK_SIZE - 1 - tid] = thread_keys[0];
}

// change it to actual parallel scan
__global__ void parallel_scan(uint64_t *blk_offset, uint64_t n) {
  for (int i = 1; i <= n; i++) {
    blk_offset[i] += blk_offset[i - 1];
  }
}

__device__ bool check_f(uint8_t *a, uint8_t *b) {

  for (int i = 0; i < char_len; i++) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

__global__ void compress_init_func(const uint8_t *const in,
                                   const uint64_t in_n_bytes,
                                   const uint64_t in_chunk_size,
                                   const uint64_t out_chunk_size,
                                   const uint64_t n_chunks, uint64_t *col_len,
                                   uint64_t *blk_offset) {

  __shared__ unsigned long long int block_len;

  if (threadIdx.x == 0) {
    block_len = 0;
    if (blockIdx.x == 0) {
      blk_offset[0] = 0;
    }
  }
  __syncthreads();

  int tid = threadIdx.x;
  int chunk_idx = blockIdx.x;

  uint64_t in_start_idx = chunk_idx * in_chunk_size;
  uint64_t out_start_idx = blk_offset[chunk_idx];
  uint64_t in_read_off = 0;
  uint8_t counter = 0;

  uint64_t used_bytes = 0;
  uint64_t consumed_bytes = 0;
  uint8_t prev_c = DEFAULT_CHAR;

  uint64_t out_len = 1;

  uint64_t in_off = 0;
  uint8_t in_flag = 0;

  uint64_t out_bytes = 1;
  uint64_t col_counter = 31;

  uint64_t byte_offset = 1;
  uint64_t byte_counter = 1;
  uint64_t cur_header_byte_pos = 0;

  uint64_t blocks = 0;

  uint8_t col_idx;
  uint8_t header_byte = 0;

  uint64_t mychunk_size = (in_chunk_size / BLK_SIZE);

  uint64_t pcounter = 0;
  uint64_t reg_counter = 0;

  uint8_t v_buffer[char_len];
  uint8_t prev_buffer[char_len];

  while (used_bytes < mychunk_size) {

    for (int i = 0; i < char_len; i++) {
      v_buffer[i] = in[in_start_idx + used_bytes + in_off + tid * 4];

      in_flag++;
      if (in_flag == 4) {
        in_off += 31 * 4;
        in_flag = 0;
      }
      used_bytes++;
    }

    if (counter == 0) {
      for (int i = 0; i < char_len; i++) {
        prev_buffer[i] = v_buffer[i];
      }
    }

    if (check_f(prev_buffer, v_buffer)) {
      counter++;
    }

    else {
      if (counter >= 3) {
        out_len += 5;
        counter = 1;

        for (int i = 0; i < char_len; i++) {
          prev_buffer[i] = v_buffer[i];
        }

        pcounter++;

        if ((++blocks) == 8) {
          out_len++;
          blocks = 0;
        }
      }

      else {

        for (uint8_t j = 0; j < counter; j++) {
          out_len += 4;
          reg_counter++;

          if ((++blocks) == 8) {
            out_len++;
            blocks = 0;
          }
        }
        counter = 1;
        for (int i = 0; i < char_len; i++) {
          prev_buffer[i] = v_buffer[i];
        }
      }
    }
  }

  if (counter >= 3) {
    pcounter++;
    out_len += 5;
  } else {
    for (uint8_t j = 0; j < counter; j++) {
      out_len += 4;
      reg_counter++;

      if ((++blocks) == 8) {
        out_len++;
        blocks = 0;
      }
    }
  }

  col_len[BLK_SIZE * chunk_idx + tid] = out_len;

  uint64_t out_len_4B = ((out_len + 3) / 4) * 4;
  atomicAdd((unsigned long long int *)&block_len,
            (unsigned long long int)out_len_4B);
  __syncthreads();
  if (threadIdx.x == 0) {
    // 128B alignment
    block_len = ((block_len + 127) / 128) * 128;
    blk_offset[chunk_idx + 1] = (uint64_t)block_len;
  }
}

__device__ void writeupdate(uint64_t *byte_counter_p, uint64_t *byte_offset_p,
                            uint64_t *col_len, uint64_t *col_counter_p,
                            int chunk_idx) {

  byte_counter_p[0]++;
  byte_offset_p[0]++;

  if ((byte_counter_p[0] % 4 == 0) && (col_counter_p[0] != 0)) {
    while ((byte_counter_p[0] >
            ((col_len[chunk_idx * BLK_SIZE + col_counter_p[0]] + 3) / 4) * 4) &&
           (col_counter_p[0] > 0)) {
      col_counter_p[0]--;
    }
    byte_offset_p[0] += col_counter_p[0] * 4;
  }
}

__global__ void compress_func(const uint8_t *const in, uint8_t *out,
                              const uint64_t in_n_bytes, uint64_t *out_n_bytes,
                              const uint64_t in_chunk_size,
                              const uint64_t n_chunks, uint64_t *col_len,
                              uint8_t *col_map, uint64_t *blk_offset) {

  int tid = threadIdx.x;
  int chunk_idx = blockIdx.x;

  uint64_t in_start_idx = chunk_idx * in_chunk_size;
  uint64_t out_start_idx = blk_offset[chunk_idx];
  uint64_t in_read_off = 0;
  uint8_t counter = 0;

  uint64_t used_bytes = 0;
  uint64_t consumed_bytes = 0;
  uint8_t prev_c = DEFAULT_CHAR;

  uint64_t out_len = 1;

  uint64_t in_off = 0;
  uint8_t in_flag = 0;

  uint64_t out_bytes = 1;
  uint64_t col_counter = 31;

  uint64_t byte_offset = 1;
  uint64_t byte_counter = 1;
  uint64_t cur_header_byte_pos = 0;

  uint64_t blocks = 0;

  uint8_t col_idx;
  uint8_t header_byte = 0;

  uint64_t mychunk_size = (in_chunk_size / BLK_SIZE);

  uint8_t v_buffer[char_len];
  uint8_t prev_buffer[char_len];

  for (int i = 0; i < 32; i++) {
    if (tid == col_map[BLK_SIZE * chunk_idx + i]) {
      col_idx = i;
    }
  }

  while (used_bytes < mychunk_size) {

    for (int i = 0; i < char_len; i++) {
      v_buffer[i] = in[in_start_idx + used_bytes + in_off + tid * 4];

      in_flag++;
      if (in_flag == 4) {
        in_off += 31 * 4;
        in_flag = 0;
      }
      used_bytes++;
    }

    if (counter == 0) {
      for (int i = 0; i < char_len; i++) {
        prev_buffer[i] = v_buffer[i];
      }
    }

    if (check_f(prev_buffer, v_buffer)) {
      counter++;
    }

    else {

      if (counter >= 3) {

        out[out_start_idx + byte_offset + col_idx * 4] = counter;

        writeupdate(&byte_counter, &byte_offset, col_len, &col_counter,
                    chunk_idx);

        for (int i = 0; i < char_len; i++) {

          out[out_start_idx + byte_offset + col_idx * 4] = prev_buffer[i];

          writeupdate(&byte_counter, &byte_offset, col_len, &col_counter,
                      chunk_idx);
        }

        counter = 1;

        for (int i = 0; i < char_len; i++) {
          prev_buffer[i] = v_buffer[i];
        }

        if ((++blocks) == 8) {

          out[out_start_idx + cur_header_byte_pos + col_idx * 4] = header_byte;
          header_byte = 0;

          cur_header_byte_pos = byte_offset;

          writeupdate(&byte_counter, &byte_offset, col_len, &col_counter,
                      chunk_idx);

          blocks = 0;
        }

        if (blocks != 0) {
          out[out_start_idx + cur_header_byte_pos + col_idx * 4] = header_byte;
        }
      }

      else {
        for (int j = 0; j < counter; j++) {
          header_byte = (header_byte | (1 << (blocks)));

          for (int i = 0; i < char_len; i++) {
            out[out_start_idx + byte_offset + col_idx * 4] = prev_buffer[i];

            writeupdate(&byte_counter, &byte_offset, col_len, &col_counter,
                        chunk_idx);
          }

          if ((++blocks) == 8) {

            out[out_start_idx + cur_header_byte_pos + col_idx * 4] =
                header_byte;
            header_byte = 0;

            cur_header_byte_pos = byte_offset;
            writeupdate(&byte_counter, &byte_offset, col_len, &col_counter,
                        chunk_idx);

            blocks = 0;
          }
          if (blocks != 0) {
            out[out_start_idx + cur_header_byte_pos + col_idx * 4] =
                header_byte;
          }
        }

        counter = 1;

        for (int i = 0; i < char_len; i++) {
          prev_buffer[i] = v_buffer[i];
        }
      }
    }
  }

  // fix it
  if (counter >= 3) {
    out[out_start_idx + byte_offset + col_idx * 4] = counter;
    writeupdate(&byte_counter, &byte_offset, col_len, &col_counter, chunk_idx);

    for (int i = 0; i < char_len; i++) {
      out[out_start_idx + byte_offset + col_idx * 4] = prev_buffer[i];
      writeupdate(&byte_counter, &byte_offset, col_len, &col_counter,
                  chunk_idx);
    }

  } else {

    for (uint8_t j = 0; j < counter; j++) {
      header_byte = (header_byte | (1 << (blocks)));

      for (int i = 0; i < char_len; i++) {

        out[out_start_idx + byte_offset + col_idx * 4] = prev_buffer[i];

        writeupdate(&byte_counter, &byte_offset, col_len, &col_counter,
                    chunk_idx);
      }

      if ((++blocks) == 8) {

        out[out_start_idx + cur_header_byte_pos + col_idx * 4] = header_byte;
        header_byte = 0;

        cur_header_byte_pos = byte_offset;

        writeupdate(&byte_counter, &byte_offset, col_len, &col_counter,
                    chunk_idx);
        blocks = 0;
      }
      if (blocks != 0) {
        out[out_start_idx + cur_header_byte_pos + col_idx * 4] = header_byte;
      }
    }
  }
}

__host__ void compress_gpu(const uint8_t *const in, uint8_t **out,
                           const uint64_t in_n_bytes, uint64_t *out_n_bytes) {
  uint8_t *d_in;
  int8_t *d_out;
  uint8_t *temp;

  uint64_t padded_in_n_bytes =
      in_n_bytes; // + (CHUNK_SIZE-(in_n_bytes % CHUNK_SIZE));
  uint32_t n_chunks = padded_in_n_bytes / CHUNK_SIZE;
  uint32_t chunk_size = padded_in_n_bytes / n_chunks;
  assert((chunk_size % READ_UNITS) == 0);
  uint64_t exp_out_chunk_size = (chunk_size + OVERHEAD_PER_CHUNK_(chunk_size));
  uint64_t exp_data_out_bytes = (n_chunks * exp_out_chunk_size);
  uint64_t len_bytes = (n_chunks * sizeof(uint64_t));
  uint64_t head_bytes = HEAD_INTS * sizeof(uint32_t);
  uint64_t out_bytes = head_bytes +        // header
                       len_bytes +         // lens
                       exp_data_out_bytes; // data

  printf("in bytes: %llu\n", in_n_bytes);

  uint64_t num_chunk = in_n_bytes / CHUNK_SIZE;
  // printf("cpu num chunk: %llu\n", num_chunk);

  // cpu
  uint8_t *cpu_data_out = (uint8_t *)malloc(exp_data_out_bytes);
  uint64_t *col_len =
      (uint64_t *)malloc(sizeof(uint64_t) * BLK_SIZE * num_chunk);
  uint8_t *col_map = (uint8_t *)malloc(BLK_SIZE * num_chunk);
  uint64_t *blk_offset = (uint64_t *)malloc(8 * (num_chunk + 1));
  uint64_t *chunk_offset = (uint64_t *)malloc(8 * (num_chunk + 1));
  uint64_t *col_offset = (uint64_t *)malloc(8 * (BLK_SIZE * num_chunk + 1));

  uint64_t *d_blk_offset;
  uint64_t *d_col_len;
  uint8_t *d_col_map;
  uint64_t *d_col_len_sorted;

  cuda_err_chk(cudaMalloc(&d_in, padded_in_n_bytes));
  cuda_err_chk(cudaMalloc(&d_col_len, sizeof(uint64_t) * BLK_SIZE * num_chunk));
  cuda_err_chk(
      cudaMalloc(&d_col_len_sorted, sizeof(uint64_t) * BLK_SIZE * num_chunk));

  cuda_err_chk(cudaMalloc(&d_col_map, BLK_SIZE * num_chunk));
  cuda_err_chk(cudaMalloc(&d_blk_offset, 8 * (num_chunk + 1)));

  cuda_err_chk(cudaMemcpy(d_in, in, in_n_bytes, cudaMemcpyHostToDevice));

  rlev1_compress_func_init<uint8_t, uint32_t><<<n_chunks, dim3(BLK_SIZE,2,1)>>>(
      d_in, chunk_size, n_chunks, d_col_len, d_col_map, d_blk_offset);


  cuda_err_chk(cudaDeviceSynchronize());

  /*compress_init_func<<<n_chunks, BLK_SIZE>>>(d_in, in_n_bytes, chunk_size,
                                             exp_out_chunk_size, n_chunks,
                                             d_col_len, d_blk_offset);


  cuda_err_chk(cudaDeviceSynchronize());
  */

  parallel_scan<<<1, 1>>>(d_blk_offset, n_chunks);
  cuda_err_chk(cudaDeviceSynchronize());
  ExampleKernel<<<n_chunks, BLK_SIZE>>>(d_col_len, d_col_map, d_col_len_sorted);
  cuda_err_chk(cudaDeviceSynchronize());

  cuda_err_chk(cudaMemcpy(col_len, d_col_len_sorted,
                          sizeof(uint64_t) * BLK_SIZE * num_chunk,
                          cudaMemcpyDeviceToHost));
  cuda_err_chk(cudaMemcpy(blk_offset, d_blk_offset, 8 * (num_chunk + 1),
                          cudaMemcpyDeviceToHost));
  cuda_err_chk(cudaMemcpy(col_map, d_col_map, BLK_SIZE * num_chunk,
                          cudaMemcpyDeviceToHost));

  uint64_t final_out_size = blk_offset[num_chunk];
  printf("first blkock offset:%llu\n", blk_offset[1]);

  printf("final out size: %llu\n", final_out_size);

  *out = new uint8_t[final_out_size];
  

  cuda_err_chk(cudaMalloc(&d_out, final_out_size));

  //compress_func<<<n_chunks, BLK_SIZE>>>(d_in, d_out, in_n_bytes, out_n_bytes,
  //                                     chunk_size, n_chunks, d_col_len_sorted,
  //                                     d_col_map, d_blk_offset);
  
  cuda_err_chk(cudaDeviceSynchronize());



rlev1_compress_func <uint8_t, uint32_t> <<< n_chunks, dim3(BLK_SIZE,2,1)>>> (d_in, d_out, chunk_size, n_chunks, d_col_len_sorted, d_col_map, d_blk_offset);

  cuda_err_chk(cudaDeviceSynchronize());



  cuda_err_chk(
      cudaMemcpy((*out), d_out, final_out_size, cudaMemcpyDeviceToHost));

  std::ofstream col_len_file("./input_data/col_len.bin", std::ofstream::binary);
  col_len_file.write((const char *)(col_len), BLK_SIZE * num_chunk * 8);
  col_len_file.close();

  std::ofstream blk_off_file("./input_data/blk_offset.bin",
                             std::ofstream::binary);
  blk_off_file.write((const char *)(blk_offset), (num_chunk + 1) * 8);
  blk_off_file.close();

  std::ofstream col_map_file("./input_data/col_map.bin", std::ofstream::binary);
  col_map_file.write((const char *)(col_map), BLK_SIZE * num_chunk);
  col_map_file.close();


  *out_n_bytes = final_out_size;
  cuda_err_chk(cudaFree(d_out));
  cuda_err_chk(cudaFree(d_col_len));
  cuda_err_chk(cudaFree(d_col_map));
  cuda_err_chk(cudaFree(d_col_len_sorted));
  cuda_err_chk(cudaFree(d_in));
  cuda_err_chk(cudaFree(d_blk_offset));
}

__host__ void decompress_gpu(const uint8_t *const in, uint8_t **out,
                             const uint64_t in_n_bytes, uint64_t *out_n_bytes) {

  std::string file_col_len = "./input_data/col_len.bin";
  std::string file_col_map = "./input_data/col_map.bin";
  std::string file_blk_off = "./input_data/blk_offset.bin";

  const char *filename_col_len = file_col_len.c_str();
  const char *filename_col_map = file_col_map.c_str();
  const char *filename_blk_off = file_blk_off.c_str();

  int fd_col_len;
  int fd_col_map;
  int fd_blk_off;

  struct stat sbcol_len;
  struct stat sbcol_map;
  struct stat sbblk_off;

  if ((fd_col_len = open(filename_col_len, O_RDONLY)) == -1) {
    printf("Fatal Error: Col Len read error\n");
    return;
  }

  if ((fd_col_map = open(filename_col_map, O_RDONLY)) == -1) {
    printf("Fatal Error: Col map read error\n");
    return;
  }

  if ((fd_blk_off = open(filename_blk_off, O_RDONLY)) == -1) {
    printf("Fatal Error: Block off read error\n");
    return;
  }

  fstat(fd_col_len, &sbcol_len);
  fstat(fd_col_map, &sbcol_map);
  fstat(fd_blk_off, &sbblk_off);

  void *map_base_col_len;
  void *map_base_col_map;
  void *map_base_blk_off;

  map_base_col_len =
      mmap(NULL, sbcol_len.st_size, PROT_READ, MAP_SHARED, fd_col_len, 0);
  map_base_col_map =
      mmap(NULL, sbcol_map.st_size, PROT_READ, MAP_SHARED, fd_col_map, 0);
  map_base_blk_off =
      mmap(NULL, sbblk_off.st_size, PROT_READ, MAP_SHARED, fd_blk_off, 0);

  uint64_t num_blk = ((uint64_t)sbblk_off.st_size / sizeof(uint64_t)) - 1;
  // uint64_t blk_size = ((uint8_t) sbcol_map.st_size / num_blk);
  uint64_t blk_size = BLK_SIZE;

  // start
  std::chrono::high_resolution_clock::time_point kernel_start =
      std::chrono::high_resolution_clock::now();

  int8_t *d_in;
  uint8_t *d_out;

  uint64_t *d_col_len;
  uint64_t *d_blk_offset;
  uint8_t *d_col_map;

  const uint8_t *const in_ = in;

  // change it later
  uint64_t in_bytes = ((uint64_t *)map_base_blk_off)[num_blk];
  uint64_t out_bytes = CHUNK_SIZE * num_blk;
  *out_n_bytes = out_bytes;

  printf("out_bytes: %llu\n", out_bytes);

  // cuda_err_chk(cudaMalloc(&d_in, in_bytes));
  // cuda_err_chk(cudaMalloc(&d_out, (*out_n_bytes)));

  cuda_err_chk(cudaMalloc(&d_in, in_bytes));
  cuda_err_chk(cudaMalloc(&d_out, (*out_n_bytes)));

  cuda_err_chk(cudaMalloc(&d_col_len, sbcol_len.st_size));
  cuda_err_chk(cudaMalloc(&d_col_map, sbcol_map.st_size));
  cuda_err_chk(cudaMalloc(&d_blk_offset, sbblk_off.st_size));

  cuda_err_chk(cudaMemcpy(d_in, in_, in_bytes, cudaMemcpyHostToDevice));

  cuda_err_chk(cudaMemcpy(d_col_len, map_base_col_len, sbcol_len.st_size,
                          cudaMemcpyHostToDevice));
  cuda_err_chk(cudaMemcpy(d_col_map, map_base_col_map, sbcol_map.st_size,
                          cudaMemcpyHostToDevice));
  cuda_err_chk(cudaMemcpy(d_blk_offset, map_base_blk_off, sbblk_off.st_size,
                          cudaMemcpyHostToDevice));

  printf("cuda malloc finished\n");
  printf("num_blk: %llu, blk_size: %llu\n", num_blk, blk_size);

  // decompress_func<<<(num_blk/4), blk_size * 4>>> (d_in, d_out, num_blk,
  // d_col_len, d_col_map, d_blk_offset);

  // real
  // decompress_func<<<(num_blk), dim3(blk_size, 3, 1)>>>(
  //    d_in, d_out, num_blk, d_col_len, d_col_map, d_blk_offset);

  // decompress_func<<<(num_blk/2), blk_size*2 >>> (d_in, d_out, num_blk,
  // d_col_len, d_col_map, d_blk_offset);



// template <typename INPUT_T, typename READ_T>
// __global__ void
// rlev1_decompress_func(const int8_t *const in, INPUT_T* out, const
//   uint64_t n_chunks, uint64_t* col_len, uint8_t *col_map, uint64_t *blk_offset) {

  uint8_t* d_out_Typed = (uint8_t*)(d_out);
  rlev1_decompress_func<uint8_t, uint32_t><<<(num_blk ), dim3(blk_size, 2, 1)>>>(d_in, d_out_Typed, num_blk, d_col_len, d_col_map, d_blk_offset);


  cudaDeviceSynchronize();
  printf("decomp function done\n");

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }

  *out = new uint8_t[out_bytes];
  cuda_err_chk(cudaMemcpy((*out), d_out, out_bytes, cudaMemcpyDeviceToHost));

  std::chrono::high_resolution_clock::time_point kernel_end =
      std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> kt =
      std::chrono::duration_cast<std::chrono::duration<double>>(kernel_end -
                                                                kernel_start);
  std::cout << "Decompression time: " << kt.count() << " secs\n";

  if (munmap(map_base_col_len, sbcol_len.st_size) == -1) {
    printf("Mem unmap error");
  }
  if (munmap(map_base_col_map, sbcol_map.st_size) == -1) {
    printf("Mem unmap error");
  }

  if (munmap(map_base_blk_off, sbblk_off.st_size) == -1) {
    printf("Mem unmap error");
  }

  close(fd_col_len);
  close(fd_blk_off);
  close(fd_col_map);

  cuda_err_chk(cudaFree(d_out));
  cuda_err_chk(cudaFree(d_in));
  cuda_err_chk(cudaFree(d_col_len));
  cuda_err_chk(cudaFree(d_col_map));
  cuda_err_chk(cudaFree(d_blk_offset));
}
} // namespace brle_trans
