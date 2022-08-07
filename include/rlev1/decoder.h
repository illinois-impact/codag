
#include <common.h>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <simt/atomic>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

// #include <common_warp.h>

#define BUFF_LEN 2

#define UNCOMP 0
#define STATIC 1
#define DYNAMIC 2
#define FULL_MASK 0xFFFFFFFF

#define MASK_4_1 0x000000FF
#define MASK_4_2 0x0000FF00
#define MASK_4_3 0x00FF0000
#define MASK_4_4 0xFF000000

#define MASK_8_1 0x0000000F
#define MASK_8_2 0x000000F0
#define MASK_8_3 0x00000F00
#define MASK_8_4 0x0000F000
#define MASK_8_5 0x000F0000
#define MASK_8_6 0x00F00000
#define MASK_8_7 0x0F000000
#define MASK_8_8 0xF0000000

template <typename T> struct decomp_write_queue_ele {
  T data;
  int header;
  bool done;
};

struct write_ele {
  int64_t value;
  int16_t delta;
  int16_t run;

  __device__ write_ele() {
    value = 0;
    delta = 0;
    run = 0;
  }

  //__device__
  // write_ele& operator =(const write_ele& a){
  //     value = a.value;
  //     delta = a.delta;
  //     run = a.run;
  // }
};

template <typename DATA_TYPE> struct decompress_output {

  DATA_TYPE *out_ptr;
  // uint64_t offset;
  uint32_t counter;

  __device__ decompress_output(uint8_t *ptr) : out_ptr((DATA_TYPE *)ptr) {
    counter = 0;
  }

  __device__ void write_value(uint8_t idx, uint64_t value) {
    if (threadIdx.x == idx) {
      out_ptr[counter] = (DATA_TYPE)value;
      counter++;
    }
  }

  __device__ void write_run(uint8_t idx, int64_t value, int16_t delta,
                            int16_t run) {

    uint64_t ptr = (uint64_t)out_ptr;
    ptr = __shfl_sync(FULL_MASK, ptr, idx);
    DATA_TYPE *out_buf = (DATA_TYPE *)ptr;

    uint32_t idx_count = __shfl_sync(FULL_MASK, counter, idx);

    idx_count += threadIdx.x;

#pragma unroll
    for (uint64_t i = threadIdx.x; i < run; i += 32, idx_count += 32) {
      int64_t out_ele = value + static_cast<int64_t>(i) * delta;
      out_buf[idx_count] = static_cast<DATA_TYPE>(out_ele);
    }

    if (threadIdx.x == idx)
      counter += run;
  }
};

template <typename COMP_COL_TYPE>
//__forceinline__
__device__ void reader_warp(decompress_input<COMP_COL_TYPE> &in,
                            queue<COMP_COL_TYPE> &rq) {
  while (true) {
    COMP_COL_TYPE v;
    int8_t rc = in.comp_read_data(FULL_MASK, &v);
    // int8_t rc = comp_read_data2(FULL_MASK, &v, in);

    if (rc == -1)
      break;
    else if (rc > 0) {

      rq.enqueue(&(v));
      // comp_enqueue<COMP_COL_TYPE, READ_COL_TYPE>(&v, &rq);
    }
  }
}

template <typename COMP_COL_TYPE, uint8_t NUM_SUBCHUNKS>
//__forceinline__
__device__ void reader_warp_orig(decompress_input<COMP_COL_TYPE> &in,
                                 queue<COMP_COL_TYPE> &rq,
                                 uint8_t active_chunks) {
  // iterate number of chunks for the single reader warp
  int t = 0;
  while (true) {
    bool done = true;
    for (uint8_t cur_chunk = 0; cur_chunk < active_chunks; cur_chunk++) {
      COMP_COL_TYPE v;
      uint8_t rc =
          comp_read_data_seq<COMP_COL_TYPE>(FULL_MASK, &v, in, cur_chunk);
      if (rc != 0)
        done = false;

      rq.warp_enqueue(&v, cur_chunk, rc);
    }
    __syncwarp();
    if (done)
      break;
  }
}

template <typename COMP_COL_TYPE, typename DATA_TYPE, size_t in_buff_len = 4>
//__forceinline__
__device__ void decoder_warp(input_stream<COMP_COL_TYPE, in_buff_len> &s,
                             decompress_output<DATA_TYPE> &out,
                             uint64_t CHUNK_SIZE, DATA_TYPE *out_buf,
                             int COMP_COL_LEN) {

  int test_idx = 40;

  uint32_t input_data_out_size = 0;
  uint64_t num_iterations = (CHUNK_SIZE / 32);

  uint64_t words_in_line = COMP_COL_LEN / sizeof(DATA_TYPE);
  uint64_t out_offset = words_in_line * threadIdx.x;
  uint64_t c = 0;

  while (input_data_out_size < num_iterations) {

    // need to read a header
    int32_t temp_byte = 0;
    s.template fetch_n_bits<int32_t>(8, &temp_byte);

    int8_t head_byte = (int8_t)(temp_byte & 0x00FF);

    // literals
    if (head_byte < 0) {
      uint64_t remaining = static_cast<uint64_t>(-head_byte);
      // if(threadIdx.x == test_idx && blockIdx.x == 1 ) printf("num_iterations:
      // %llu remaining: %x\n", num_iterations, remaining);

      for (uint64_t i = 0; i < remaining; ++i) {

        DATA_TYPE value = 0;
        int64_t offset = 0;

        // read var-int value
        bool read_next = true;
        while (read_next) {
          temp_byte = 0;
          s.template fetch_n_bits<int32_t>(8, &temp_byte);
          int8_t in_data = 0;
          in_data = (int8_t)(in_data | (temp_byte & 0x00FF));

          if (in_data >= 0) {
            value |= (static_cast<DATA_TYPE>(in_data) << offset);
            read_next = false;
          } else {
            value |= ((static_cast<DATA_TYPE>(in_data) & 0x7f) << offset);
            offset += 7;
          }
        }

        decomp_write_queue_ele<DATA_TYPE> qe;
        qe.data = value;

        out_buf[out_offset] = value;

        // if(threadIdx.x == 0 && blockIdx.x == 0) printf(" value: %lx\n",
        // (unsigned long) value);

        c++;
        if (c == words_in_line) {
          out_offset += words_in_line * 31;
          c = 0;
        }

        out_offset++;

        // mq.enqueue(&qe);
        //(threadIdx.x == 0 && blockIdx.x == 0 && input_data_out_size <= 20)
        //printf("out: %c\n", value);

        input_data_out_size += sizeof(DATA_TYPE);
      }

    }
    // compresssed data
    else {
      uint64_t remaining = static_cast<uint64_t>(head_byte);

      temp_byte = 0;
      s.template fetch_n_bits<int32_t>(8, &temp_byte);
      int8_t delta = (int8_t)(temp_byte & 0x00FF);

      DATA_TYPE value = 0;
      int64_t offset = 0;

      int32_t in_data;

      while (1) {
        temp_byte = 0;
        s.template fetch_n_bits<int32_t>(8, &temp_byte);

        int8_t in_data = (int8_t)(temp_byte & 0x00FF);

        if (in_data >= 0) {
          // if(threadIdx.x == test_idx && blockIdx.x == 0) printf("data4:
          // %x\n", in_data);

          value |= (static_cast<DATA_TYPE>(in_data) << offset);
          break;
        } else {
          // if(threadIdx.x == test_idx && blockIdx.x == 0) printf("data5:
          // %x\n", in_data);

          value |= ((static_cast<DATA_TYPE>(in_data) & 0x7f) << offset);
          offset += 7;
        }
      }

      //   if(threadIdx.x == 0 && blockIdx.x == 0) printf("comp value: %lx\n",
      //   (unsigned long) value);

      // decoding the compresssed stream
      for (uint64_t i = 0; i < remaining + 3; ++i) {
        int64_t out_ele = value + static_cast<int64_t>(i) * delta;
        // write out_ele

        // temp_write_word = temp_write_word | (static_cast<READ_T>(out_ele) <<
        // (temp_word_count * sizeof(INPUT_T) * 8));

        decomp_write_queue_ele<DATA_TYPE> qe;
        qe.data = static_cast<DATA_TYPE>(out_ele);

        out_buf[out_offset] = static_cast<DATA_TYPE>(out_ele);
        c++;
        if (c == words_in_line) {
          out_offset += words_in_line * 31;
          c = 0;
        }

        out_offset++;

        // mq.enqueue(&qe);

        input_data_out_size += sizeof(DATA_TYPE);
      }
    }
  }

  // printf("yeah done!! bid:%i tid: %i\n", blockIdx.x, threadIdx.x);
}

// only one thread in a warp is decoding
template <typename COMP_COL_TYPE, typename DATA_TYPE, size_t in_buff_len = 4>
__device__ void
decoder_warp_orig_dw(input_stream<COMP_COL_TYPE, in_buff_len> &s,
                     uint64_t CHUNK_SIZE, DATA_TYPE *out_buf) {

  uint32_t input_data_out_size = 0;
  uint64_t out_offset = 0;

  while (input_data_out_size < CHUNK_SIZE) {

    // need to read a header
    int32_t temp_byte = 0;
    if (threadIdx.x == 0)
      s.template fetch_n_bits<int32_t>(8, &temp_byte);

    int8_t head_byte = (int8_t)(temp_byte & 0x00FF);
    head_byte = __shfl_sync(FULL_MASK, head_byte, 0);

    // literals
    if (head_byte < 0) {
      uint64_t remaining = static_cast<uint64_t>(-head_byte);

      for (uint64_t i = 0; i < remaining; ++i) {

        DATA_TYPE value = 0;
        int64_t offset = 0;

        // read var-int value
        bool read_next = true & (threadIdx.x == 0);
        while (read_next) {
          temp_byte = 0;
          s.template fetch_n_bits<int32_t>(8, &temp_byte);
          int8_t in_data = 0;
          in_data = (int8_t)(in_data | (temp_byte & 0x00FF));

          if (in_data >= 0) {
            value |= (static_cast<DATA_TYPE>(in_data) << offset);
            read_next = false;
          } else {
            value |= ((static_cast<DATA_TYPE>(in_data) & 0x7f) << offset);
            offset += 7;
          }
        }

        if (threadIdx.x == 0)
          out_buf[out_offset] = value;

        out_offset++;
        input_data_out_size += sizeof(DATA_TYPE);
      }

    }
    // compresssed data
    else {
      uint64_t remaining = static_cast<uint64_t>(head_byte);

      temp_byte = 0;
      if (threadIdx.x == 0)
        s.template fetch_n_bits<int32_t>(8, &temp_byte);
      int8_t delta = (int8_t)(temp_byte & 0x00FF);
      delta = __shfl_sync(FULL_MASK, delta, 0);

      DATA_TYPE value = 0;
      int64_t offset = 0;

      int32_t in_data;

      while (threadIdx.x == 0) {
        temp_byte = 0;
        s.template fetch_n_bits<int32_t>(8, &temp_byte);

        int8_t in_data = (int8_t)(temp_byte & 0x00FF);

        if (in_data >= 0) {

          value |= (static_cast<DATA_TYPE>(in_data) << offset);
          break;
        } else {

          value |= ((static_cast<DATA_TYPE>(in_data) & 0x7f) << offset);
          offset += 7;
        }
      }

      value = __shfl_sync(FULL_MASK, value, 0);

      uint64_t next_out_offset = out_offset + remaining + 3;
      out_offset += min((uint64_t)threadIdx.x, remaining + 3);

      // decoding the compresssed stream
      for (uint64_t i = threadIdx.x; i < remaining + 3;
           i += 32, out_offset += 32) {

        int64_t out_ele = value + static_cast<int64_t>(i) * delta;
        out_buf[out_offset] = static_cast<DATA_TYPE>(out_ele);
      }

      out_offset = next_out_offset;
      input_data_out_size += (sizeof(DATA_TYPE) * (remaining + 3));
    }
  }
}

// only one thread in a warp is decoding
template <typename COMP_COL_TYPE, typename DATA_TYPE, size_t in_buff_len = 4>
__device__ void
decoder_warp_orig_dw_multi(input_stream<COMP_COL_TYPE, in_buff_len> &s,
                           uint64_t CHUNK_SIZE, DATA_TYPE *out_buf) {

  uint32_t input_data_out_size = 0;
  uint64_t out_offset = 0;

  while (input_data_out_size < CHUNK_SIZE) {

    // need to read a header
    int32_t temp_byte = 0;
    s.template fetch_n_bits<int32_t>(8, &temp_byte);

    int8_t head_byte = (int8_t)(temp_byte & 0x00FF);
    // head_byte = __shfl_sync(FULL_MASK, head_byte, 0);

    // literals
    if (head_byte < 0) {
      uint64_t remaining = static_cast<uint64_t>(-head_byte);

      for (uint64_t i = 0; i < remaining; ++i) {

        DATA_TYPE value = 0;
        int64_t offset = 0;

        // read var-int value
        bool read_next = true;
        while (read_next) {
          temp_byte = 0;
          s.template fetch_n_bits<int32_t>(8, &temp_byte);
          int8_t in_data = 0;
          in_data = (int8_t)(in_data | (temp_byte & 0x00FF));

          if (in_data >= 0) {
            value |= (static_cast<DATA_TYPE>(in_data) << offset);
            read_next = false;
          } else {
            value |= ((static_cast<DATA_TYPE>(in_data) & 0x7f) << offset);
            offset += 7;
          }
        }

        out_buf[out_offset] = value;

        out_offset++;
        input_data_out_size += sizeof(DATA_TYPE);
      }

    }
    // compresssed data
    else {
      uint64_t remaining = static_cast<uint64_t>(head_byte);

      temp_byte = 0;
      s.template fetch_n_bits<int32_t>(8, &temp_byte);
      int8_t delta = (int8_t)(temp_byte & 0x00FF);
      // delta = __shfl_sync(FULL_MASK, delta, 0);

      DATA_TYPE value = 0;
      int64_t offset = 0;

      int32_t in_data;

      while (1) {
        temp_byte = 0;
        s.template fetch_n_bits<int32_t>(8, &temp_byte);

        int8_t in_data = (int8_t)(temp_byte & 0x00FF);

        if (in_data >= 0) {

          value |= (static_cast<DATA_TYPE>(in_data) << offset);
          break;
        } else {

          value |= ((static_cast<DATA_TYPE>(in_data) & 0x7f) << offset);
          offset += 7;
        }
      }

      for (uint64_t i = 0; i < remaining + 3; ++i) {
        int64_t out_ele = value + static_cast<int64_t>(i) * delta;
        out_buf[out_offset] = static_cast<DATA_TYPE>(out_ele);
        out_offset++;
      }

      input_data_out_size += (sizeof(DATA_TYPE) * (remaining + 3));
    }
  }
}

// only one thread in a warp is decoding
template <typename COMP_COL_TYPE, typename DATA_TYPE, size_t in_buff_len = 4>
__device__ void
decoder_warp_orig_rdw(full_warp_input_stream<COMP_COL_TYPE, in_buff_len> &s,
                      uint64_t CHUNK_SIZE, DATA_TYPE *out_buf) {

  uint32_t input_data_out_size = 0;
  uint64_t out_offset = 0;

  while (input_data_out_size < CHUNK_SIZE) {

    // need to read a header
    int32_t temp_byte = 0;
    s.template fetch_n_bits<int32_t>(8, &temp_byte);

    int8_t head_byte = (int8_t)(temp_byte & 0x00FF);
    // head_byte = __shfl_sync(FULL_MASK, head_byte, 0);

    // literals
    if (head_byte < 0) {
      uint64_t remaining = static_cast<uint64_t>(-head_byte);

      for (uint64_t i = 0; i < remaining; ++i) {

        DATA_TYPE value = 0;
        int64_t offset = 0;

        // read var-int value
        // bool read_next = true & (threadIdx.x == 0);
        bool read_next = true;

        while (read_next) {
          temp_byte = 0;
          s.template fetch_n_bits<int32_t>(8, &temp_byte);

          int8_t in_data = 0;
          in_data = (int8_t)(in_data | (temp_byte & 0x00FF));

          if (in_data >= 0) {
            value |= (static_cast<DATA_TYPE>(in_data) << offset);
            read_next = false;
          } else {
            value |= ((static_cast<DATA_TYPE>(in_data) & 0x7f) << offset);
            offset += 7;
          }
        }

        if (threadIdx.x == 0)
          out_buf[out_offset] = value;

        out_offset++;
        input_data_out_size += sizeof(DATA_TYPE);
      }

    }
    // compresssed data
    else {
      uint64_t remaining = static_cast<uint64_t>(head_byte);

      temp_byte = 0;
      s.template fetch_n_bits<int32_t>(8, &temp_byte);

      int8_t delta = (int8_t)(temp_byte & 0x00FF);
      // delta = __shfl_sync(FULL_MASK, delta, 0);

      DATA_TYPE value = 0;
      int64_t offset = 0;

      int32_t in_data;

      while (1) {
        temp_byte = 0;
        s.template fetch_n_bits<int32_t>(8, &temp_byte);

        int8_t in_data = (int8_t)(temp_byte & 0x00FF);

        if (in_data >= 0) {

          value |= (static_cast<DATA_TYPE>(in_data) << offset);
          break;
        } else {

          value |= ((static_cast<DATA_TYPE>(in_data) & 0x7f) << offset);
          offset += 7;
        }
      }

      // value = __shfl_sync(FULL_MASK, value, 0);

      uint64_t next_out_offset = out_offset + remaining + 3;
      out_offset += min((uint64_t)threadIdx.x, remaining + 3);

      // decoding the compresssed stream
      for (uint64_t i = threadIdx.x; i < remaining + 3;
           i += 32, out_offset += 32) {

        int64_t out_ele = value + static_cast<int64_t>(i) * delta;
        out_buf[out_offset] = static_cast<DATA_TYPE>(out_ele);
      }

      out_offset = next_out_offset;
      input_data_out_size += (sizeof(DATA_TYPE) * (remaining + 3));
    }
  }
}

// only one thread in a warp is decoding
template <typename COMP_COL_TYPE, typename DATA_TYPE, size_t in_buff_len = 4>
__device__ void
decoder_warp_orig_rdw_fsm(full_warp_input_stream<COMP_COL_TYPE, in_buff_len> &s,
                          uint64_t CHUNK_SIZE, DATA_TYPE *out_buf) {

  // printf("CHUNK_SIZE: %llu\n", CHUNK_SIZE);

  bool track = false;
  bool out_track = false;
  bool size_track = false;

  uint32_t input_data_out_size = 0;
  uint64_t out_offset = 0;
  bool read_next;
  int32_t temp_byte = 0;
  int8_t head_byte;
  uint64_t remaining;
  int8_t delta;
  DATA_TYPE value;
  int64_t offset;
  int8_t in_data;
  uint64_t next_out_offset;
  int i = 0;
  // FSM
  char state = 'R';
  int Decode_State = 0;

  while (1) {
  FSM:
    state = __shfl_sync(FULL_MASK, state, 0);

    switch (state) {
    case 'R': {
      // fill buffer
      s.on_demand_read();

      if (threadIdx.x == 0) {
        switch (Decode_State) {
        case 0:
          goto DECODE_0;
        case 1:
          goto DECODE_1;
        case 2:
          goto DECODE_2;
        case 3:
          goto DECODE_3;
        case 4:
          goto DECODE_4;
        default:
          goto DONE;
        }
      }
      break;
    }

    case 'W': {
      remaining = __shfl_sync(FULL_MASK, remaining, 0);
      out_offset = __shfl_sync(FULL_MASK, out_offset, 0);
      delta = __shfl_sync(FULL_MASK, delta, 0);
      value = __shfl_sync(FULL_MASK, value, 0);

      for (i = 0; i < remaining + 3; i++) {
        int64_t out_ele = value + static_cast<int64_t>(i) * delta;
        out_buf[out_offset] = static_cast<DATA_TYPE>(out_ele);
        out_offset++;
      }

      if (threadIdx.x == 0)
        goto W_DONE;
      break;
    }

    case 'D':
      goto DONE;
    default:
      goto DONE;
    }
  }

DECODE_0:
  if (blockIdx.x == 0 && track)
    printf("decode start\n");

  while (input_data_out_size < CHUNK_SIZE) {
    // need to read a header
    temp_byte = 0;
  DECODE_1:
    if (s.template fetch_n_bits_single<int32_t>(8, &temp_byte) == false) {
      if (blockIdx.x == 0 && track)
        printf("D1 fetch\n");

      state = 'R';
      Decode_State = 1;
      goto FSM;
    }

    head_byte = (int8_t)(temp_byte & 0x00FF);

    if (head_byte < 0) {
      remaining = static_cast<uint64_t>(-head_byte);

      for (i = 0; i < remaining; ++i) {
        value = 0;
        offset = 0;
        read_next = true;

        while (read_next) {
        DECODE_2:

          temp_byte = 0;
          if (s.template fetch_n_bits_single<int32_t>(8, &temp_byte) == false) {
            if (blockIdx.x == 0 && track)
              printf("read\n");

            state = 'R';
            Decode_State = 2;
            goto FSM;
          }

          in_data = 0;
          in_data = (int8_t)(in_data | (temp_byte & 0x00FF));

          if (in_data >= 0) {
            value |= (static_cast<DATA_TYPE>(in_data) << offset);
            read_next = false;
          } else {
            value |= ((static_cast<DATA_TYPE>(in_data) & 0x7f) << offset);
            offset += 7;
          }
        }

        out_buf[out_offset] = value;
        if (blockIdx.x == 0 && out_track)
          printf("%lx\n", value);

        out_offset++;
        input_data_out_size += sizeof(DATA_TYPE);
        // printf("out: %llu add: %llu\n", (unsigned long long)
        // input_data_out_size, (unsigned long long)(sizeof(DATA_TYPE)) );
      }
    }

    // compresssed data
    else {
      remaining = static_cast<uint64_t>(head_byte);
    DECODE_3:
      temp_byte = 0;
      if (s.template fetch_n_bits_single<int32_t>(8, &temp_byte) == false) {

        state = 'R';
        Decode_State = 3;
        goto FSM;
      }

      delta = (int8_t)(temp_byte & 0x00FF);
      value = 0;
      offset = 0;

      while (1) {

      DECODE_4:
        temp_byte = 0;
        if (s.template fetch_n_bits_single<int32_t>(8, &temp_byte) == false) {

          state = 'R';
          Decode_State = 4;
          goto FSM;
        }

        in_data = (int8_t)(temp_byte & 0x00FF);

        if (in_data >= 0) {
          value |= (static_cast<DATA_TYPE>(in_data) << offset);
          break;
        } else {
          value |= ((static_cast<DATA_TYPE>(in_data) & 0x7f) << offset);
          offset += 7;
        }
      }

      next_out_offset = out_offset + remaining + 3;
      out_offset += min((uint64_t)threadIdx.x, remaining + 3);

      // for(i = 0; i < remaining + 3; i++){
      //     int64_t out_ele = value + static_cast<int64_t>(i) * delta;
      //     if(blockIdx.x == 0 && out_track) printf("%lx\n", out_ele);
      //     out_buf[out_offset] =  static_cast<DATA_TYPE>(out_ele);
      //     out_offset++;
      // }

      state = 'W';
      goto FSM;

      // for(i = 0; i < remaining + 3; i+=32){
      //     int64_t out_ele = value + static_cast<int64_t>(i) * delta;
      //     out_buf[out_offset] =  static_cast<DATA_TYPE>(out_ele);
      //     out_offset+=32;
      // }

    W_DONE:
      out_offset = next_out_offset;
      input_data_out_size += (sizeof(DATA_TYPE) * (remaining + 3));
    }
  }

  state = 'D';
  goto FSM;

DONE:
  return;
}

// only one thread in a warp is decoding
template <typename COMP_COL_TYPE, typename DATA_TYPE, size_t in_buff_len = 4>
__device__ void
decoder_warp_orig_multi(input_stream<COMP_COL_TYPE, in_buff_len> &s,
                        queue<write_ele> &wq, uint64_t CHUNK_SIZE,
                        DATA_TYPE *out_buf) {

  uint32_t input_data_out_size = 0;
  uint64_t out_offset = 0;

  while (input_data_out_size < CHUNK_SIZE) {

    // need to read a header
    int32_t temp_byte = 0;
    s.template fetch_n_bits<int32_t>(8, &temp_byte);

    int8_t head_byte = (int8_t)(temp_byte & 0x00FF);
    // head_byte = __shfl_sync(FULL_MASK, head_byte, 0);

    // literals
    if (head_byte < 0) {
      uint64_t remaining = static_cast<uint64_t>(-head_byte);

      for (uint64_t i = 0; i < remaining; ++i) {

        DATA_TYPE value = 0;
        int64_t offset = 0;

        // read var-int value
        bool read_next = true;
        while (read_next) {
          temp_byte = 0;
          s.template fetch_n_bits<int32_t>(8, &temp_byte);
          int8_t in_data = 0;
          in_data = (int8_t)(in_data | (temp_byte & 0x00FF));

          if (in_data >= 0) {
            value |= (static_cast<DATA_TYPE>(in_data) << offset);
            read_next = false;
          } else {
            value |= ((static_cast<DATA_TYPE>(in_data) & 0x7f) << offset);
            offset += 7;
          }
        }

        write_ele w_ele;
        w_ele.run = 0;
        w_ele.value = value;

        wq.enqueue(&w_ele);

        // out_buf[out_offset] = value;
        // out_offset++;
        input_data_out_size += sizeof(DATA_TYPE);
      }

    }
    // compresssed data
    else {
      uint64_t remaining = static_cast<uint64_t>(head_byte);

      temp_byte = 0;
      s.template fetch_n_bits<int32_t>(8, &temp_byte);
      int8_t delta = (int8_t)(temp_byte & 0x00FF);
      // delta = __shfl_sync(FULL_MASK, delta, 0);

      DATA_TYPE value = 0;
      int64_t offset = 0;

      int32_t in_data;

      while (1) {
        temp_byte = 0;
        s.template fetch_n_bits<int32_t>(8, &temp_byte);

        int8_t in_data = (int8_t)(temp_byte & 0x00FF);

        if (in_data >= 0) {

          value |= (static_cast<DATA_TYPE>(in_data) << offset);
          break;
        } else {

          value |= ((static_cast<DATA_TYPE>(in_data) & 0x7f) << offset);
          offset += 7;
        }
      }

      write_ele w_ele;
      w_ele.run = remaining + 3;
      w_ele.value = value;
      w_ele.delta = delta;

      wq.enqueue(&w_ele);

      // for(uint64_t i = 0; i < remaining + 3; ++i){
      //     int64_t out_ele = value + static_cast<int64_t>(i) * delta;
      //     out_buf[out_offset] =  static_cast<DATA_TYPE>(out_ele);
      //     out_offset++;
      // }

      input_data_out_size += (sizeof(DATA_TYPE) * (remaining + 3));
    }
  }
}

template <typename COMP_COL_TYPE, typename DATA_TYPE, typename OUT_COL_TYPE,
          int NUM_CHUNKS, uint16_t queue_size = 4, int NT = 64, int BT = 32>
__global__ void __launch_bounds__(NT)
    inflate_orig_rdw(uint8_t *comp_ptr, uint8_t *out,
                     const uint64_t *const col_len_ptr,
                     const uint64_t *const blk_offset_ptr, uint64_t CHUNK_SIZE,
                     int COMP_COL_LEN, uint64_t num_chunks) {

  __shared__ COMP_COL_TYPE in_queue_[NUM_CHUNKS][queue_size];

  uint8_t active_chunks = NUM_CHUNKS;
  if ((blockIdx.x + 1) * NUM_CHUNKS > num_chunks) {
    active_chunks = num_chunks - blockIdx.x * NUM_CHUNKS;
  }

  int my_queue = (threadIdx.y);
  int my_block_idx = (blockIdx.x * NUM_CHUNKS + threadIdx.y);
  uint64_t col_len = col_len_ptr[my_block_idx];
  __syncthreads();

  full_warp_input_stream<COMP_COL_TYPE, queue_size> s(
      comp_ptr, col_len, blk_offset_ptr[my_block_idx] / sizeof(COMP_COL_TYPE),
      in_queue_[my_queue]);
  __syncthreads();
  unsigned int chunk_id = blockIdx.x * NUM_CHUNKS + (threadIdx.y);

  // decoder_warp_orig_rdw<COMP_COL_TYPE, DATA_TYPE, queue_size>(s, CHUNK_SIZE,
  // (DATA_TYPE*)(out + CHUNK_SIZE * chunk_id));
  decoder_warp_orig_rdw_fsm<COMP_COL_TYPE, DATA_TYPE, queue_size>(
      s, CHUNK_SIZE, (DATA_TYPE *)(out + CHUNK_SIZE * chunk_id));
}
