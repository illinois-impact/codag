
#include <common.h>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <simt/atomic>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

// #include <common_warp.h>

#define FULL_MASK 0xFFFFFFFF

struct write_queue_ele {
  int8_t data;
  int header;
  bool done;
};

template <typename DECOMP_COL_TYPE> struct compress_output {

  DECOMP_COL_TYPE *out_ptr;
  uint64_t offset;
  bool write_flag;

  __device__ compress_output(uint8_t *ptr) : out_ptr((DECOMP_COL_TYPE *)ptr) {
    offset = 0;
    write_flag = true;
  }

  __inline__ __device__ uint64_t warp_write(DECOMP_COL_TYPE data) {

    uint32_t write_sync = __ballot_sync(FULL_MASK, write_flag);
    uint64_t idx = 0;
    if (write_flag) {
      idx = offset + __popc(write_sync & (0xffffffff >> (32 - threadIdx.x)));
      out_ptr[idx] = data;

      // if(blockIdx.x == 0 && (idx * 4 == (6740))){
      //     printf("tid: %i idx: %llu data:%i\n",threadIdx.x, idx, data);
      // }

      offset += __popc(write_sync);
    }
    return idx;
  }

  __inline__ __device__ void set_flag() { write_flag = true; }

  __device__ uint64_t get_offset() { return offset; }
  __device__ void update_header(uint64_t offset, uint64_t inword_offset,
                                uint8_t head) {
    uint8_t *byte_out = (uint8_t *)out_ptr;
    byte_out[offset * sizeof(DECOMP_COL_TYPE) + inword_offset] = head;
  }
};

template <typename READ_COL_TYPE, typename DATA_TYPE>
__device__ void compression_reader_warp(queue<DATA_TYPE> &rq,
                                        DATA_TYPE *input_data_ptr,
                                        uint64_t CHUNK_SIZE, int COMP_COL_LEN) {

  uint32_t num_iterations = (CHUNK_SIZE / 32 + COMP_COL_LEN - 1) / COMP_COL_LEN;
  uint32_t num_data_per_col = COMP_COL_LEN / sizeof(DATA_TYPE);
  uint32_t offset = threadIdx.x * num_data_per_col;

  for (int iter = 0; iter < num_iterations; iter++) {
    for (int i = 0; i < num_data_per_col; i++) {
      DATA_TYPE read_data = input_data_ptr[offset + i];
      rq.enqueue(&read_data);
    }
    offset += (num_data_per_col * 32);
  }
  return;
}

template <typename DATA_TYPE>
__device__ uint64_t get_varint_size(DATA_TYPE val) {
  DATA_TYPE write_val = val;
  uint64_t out_len = 0;

  do {
    out_len++;
    write_val = write_val >> 7;
  } while (write_val != 0);

  return out_len;
}

template <typename DATA_TYPE>
__device__ void enqueue_varint(queue<write_queue_ele> &wq, DATA_TYPE val) {
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

__device__ void enqueue_byte(queue<write_queue_ele> &wq, uint8_t val) {

  write_queue_ele write_data;
  write_data.done = false;
  write_data.header = 0;
  write_data.data = val;
  wq.enqueue(&write_data);
}

__device__ void enqueue_header_holder(queue<write_queue_ele> &wq, int8_t val) {
  write_queue_ele write_data;
  write_data.done = false;
  write_data.header = 1;
  write_data.data = val;
  wq.enqueue(&write_data);
}

__device__ void enqueue_header_setting(queue<write_queue_ele> &wq, int8_t val) {
  write_queue_ele write_data;
  write_data.done = false;
  write_data.header = 2;
  write_data.data = val;
  wq.enqueue(&write_data);
}

template <typename READ_COL_TYPE, typename DATA_TYPE, size_t in_buff_len = 4>
__device__ void compression_init_warp(queue<DATA_TYPE> &rq,
                                      uint64_t *col_len_ptr,
                                      uint64_t *blk_offset_ptr,
                                      uint64_t CHUNK_SIZE, int COMP_COL_LEN) {

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

  bool reset_flag = false;

  // first data
  DATA_TYPE read_data = 0;
  rq.dequeue(&read_data);
  // if(threadIdx.x == 9 && blockIdx.x == 6) printf("readdata: %x\n",
  // read_data);

  delta_count = 1;
  prev_val = read_data;

  data_buffer[data_buffer_head] = read_data;
  data_buffer_head = (data_buffer_head + 1) % 2;
  data_buffer_count++;
  used_bytes += read_bytes;

  rq.dequeue(&read_data);
  // f(threadIdx.x == 9 && blockIdx.x == 6) printf("readdata: %x\n", read_data);

  // second data
  int64_t temp_diff = read_data - prev_val;

  if (temp_diff > 127 || temp_diff < -128) {
    delta_flag = false;
    lit_idx = out_offset;

    // enqueue_header_holder(wq, 1);
    out_len += 1;
    lit_count = 1;

    DATA_TYPE lit_val = data_buffer[0];

    out_len += get_varint_size<DATA_TYPE>(lit_val);

    data_buffer_count--;
  }

  else {
    delta_flag = true;
    cur_delta = (int8_t)temp_diff;
    // if(threadIdx.x == 17 && blockIdx.x == 7 ) printf("cur delta: %x temp_diff
    // %lx  \n",  cur_delta, temp_diff);

    delta_count++;
  }

  prev_val = read_data;
  data_buffer[data_buffer_head] = read_data;
  data_buffer_head = (data_buffer_head + 1) % 2;
  data_buffer_count++;
  used_bytes += read_bytes;

  while (used_bytes < my_chunk_size) {
    // if(threadIdx.x == 24 && blockIdx.x == 93064 ) printf("out len: %llu\n",
    // out_len);

    if (lit_count == 127) {
      lit_count = 0;
      // reset_flag = true;
      // delta_count = 1;
    }

    rq.dequeue(&read_data);
    used_bytes += read_bytes;

    if (delta_count == 1) {

      temp_diff = read_data - prev_val;
      if (temp_diff > 127 || temp_diff < -128) {
        delta_flag = false;
        if (lit_count == 0) {
          lit_idx = out_offset;
          // enqueue_header_holder(wq, 1);
          out_len += 1;
        }

        int8_t data_buffer_tail = (data_buffer_head == 0) ? 1 : 0;
        DATA_TYPE lit_val = data_buffer[data_buffer_tail];
        out_len += get_varint_size<DATA_TYPE>(lit_val);

        lit_count++;
        data_buffer_count--;
      } else {
        delta_flag = true;
        cur_delta = (int8_t)temp_diff;
        // if(threadIdx.x == 17 && blockIdx.x == 7 ) printf("cur delta3: %x
        // temp_diff %lx  \n",  cur_delta, temp_diff);

        delta_count++;
      }
      prev_val = read_data;
      data_buffer[data_buffer_head] = read_data;
      data_buffer_head = (data_buffer_head + 1) % 2;
      data_buffer_count++;

      continue;
    }

    // matched
    if (prev_val + cur_delta == read_data && delta_flag) {
      delta_count++;
      if (delta_count == 3) {
        // if(threadIdx.x == 17 && blockIdx.x == 7 ) printf("cur delta2: %x
        // temp_diff %lx  \n",  cur_delta, temp_diff);

        delta_first_val = data_buffer[data_buffer_head];

        if (lit_count != 0) {
          // if(threadIdx.x == 9 && blockIdx.x == 6) printf("set head1:");

          // enqueue_header_setting(wq, static_cast<int8_t>(-lit_count));
          // out_buffer_ptr[lit_idx] = static_cast<int8_t>(-lit_count);
          lit_count = 0;
        }
      }
      // max
      else if (delta_count == 130) {
        int8_t write_byte = delta_count - 4;
        // enqueue_byte(wq, write_byte);
        write_byte = (uint8_t)cur_delta;

        // enqueue_byte(wq, write_byte);

        out_len += 2;
        out_len += get_varint_size(delta_first_val);

        delta_count = 1;
        data_buffer_count = 0;
        lit_count = 0;
      }

      data_buffer[data_buffer_head] = read_data;
      data_buffer_head = (data_buffer_head + 1) % 2;
      data_buffer_count = min(data_buffer_count + 1, 2);
      prev_val = read_data;
    }

    // not matched

    else {
      if (delta_count >= 3) {

        int8_t write_byte = delta_count - 3;

        // enqueue_byte(wq, write_byte);

        write_byte = (uint8_t)cur_delta;

        // enqueue_byte(wq, write_byte);

        out_len += 2;
        out_len += get_varint_size<DATA_TYPE>(delta_first_val);

        // if(threadIdx.x == 17 && blockIdx.x == 7 ) printf("write_byte: %x cur
        // delta:%x  fv: %x\n", delta_count - 3, cur_delta, delta_first_val);

        delta_count = 1;
        data_buffer_count = 0;
        lit_count = 0;
        data_buffer[data_buffer_head] = read_data;
        data_buffer_head = (data_buffer_head + 1) % 2;
        data_buffer_count = min(data_buffer_count + 1, 2);
        prev_val = read_data;
      }

      else {
        if (lit_count == 0) {
          lit_idx = out_offset;
          // write_byte_op(out_buffer_ptr, &out_bytes, (uint8_t) 3, &out_offset,
          // s_col_len, &col_counter, COMP_WRITE_BYTES);
          // enqueue_header_holder(wq, 1);
          out_len++;
        }
        lit_count++;
        DATA_TYPE lit_val = data_buffer[data_buffer_head];

        // write_varint(wq, lit_val);
        out_len += get_varint_size<DATA_TYPE>(lit_val);

        if (lit_count == 127) {
          // out_buffer_ptr[lit_idx] = static_cast<int8_t>(-lit_count);
          //  if(threadIdx.x == 9 && blockIdx.x == 6) printf("set head2:");

          // enqueue_header_setting(wq, static_cast<int8_t>(-lit_count));

          lit_count = 0;
        }

        int64_t temp_diff = read_data - prev_val;
        if (temp_diff > 127 || temp_diff < -128) {

          if (lit_count == 0) {
            lit_idx = out_offset;
            // enqueue_header_holder(wq,1);
            out_len++;
            // write_byte_op(out_buffer_ptr, &out_bytes, (uint8_t) 3,
            // &out_offset,
            //                          s_col_len, &col_counter,
            //                          COMP_WRITE_BYTES);
          }

          delta_flag = false;
          data_buffer_count = 0;
          int8_t data_buffer_tail = (data_buffer_head == 0) ? 1 : 0;
          DATA_TYPE lit_val = data_buffer[data_buffer_tail];
          // enqueue_varint(wq, lit_val);
          out_len += get_varint_size<DATA_TYPE>(lit_val);

          lit_count++;
          delta_count = 1;
        } else {
          data_buffer_count = 1;
          delta_flag = true;
          cur_delta = (int8_t)temp_diff;
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
    // enqueue_byte(wq, write_byte);

    write_byte = (uint8_t)cur_delta;
    // enqueue_byte(wq, write_byte);
    out_len += 2;
    out_len += get_varint_size<DATA_TYPE>(delta_first_val);
    // if(threadIdx.x == 24 && blockIdx.x == 93064 ) printf("out len1: %llu\n",
    // out_len);

  }

  else {
    // update lit count

    if (data_buffer_count == 1) {
      if (lit_count == 127)
        lit_count = 0;

      if (lit_count == 0)
        out_len++;

      int8_t data_buffer_tail = (data_buffer_head == 0) ? 1 : 0;
      DATA_TYPE lit_val = data_buffer[data_buffer_tail];
      // enqueue_varint(wq, lit_val);
      out_len += get_varint_size<DATA_TYPE>(lit_val);
      //  if(threadIdx.x == 24 && blockIdx.x == 93064 ) printf("out len4:
      //  %llu\n", out_len);

      lit_count++;
    }

    if (data_buffer_count == 2) {

      if (lit_count == 127)
        lit_count = 0;

      if (lit_count == 0)
        out_len++;
      DATA_TYPE lit_val = data_buffer[data_buffer_head];
      // enqueue_varint(wq, lit_val);
      out_len += get_varint_size<DATA_TYPE>(lit_val);

      lit_count++;

      if (lit_count == 127)
        lit_count = 0;

      if (lit_count == 0)
        out_len++;

      data_buffer_head = (data_buffer_head + 1) % 2;
      lit_val = data_buffer[data_buffer_head];
      // enqueue_varint(wq, lit_val);
      out_len += get_varint_size<DATA_TYPE>(lit_val);
      //  if(threadIdx.x == 24 && blockIdx.x == 93064 ) printf("out len6:
      //  %llu\n", out_len);

      lit_count++;
    }
    //  if(threadIdx.x == 9 && blockIdx.x == 6) printf("set head3:");

    // enqueue_header_setting(wq, (-lit_count));
    // if(threadIdx.x == 24 && blockIdx.x == 93064 ) printf("out len2: %llu\n",
    // out_len);
  }

  col_len_ptr[threadIdx.x + 32 * blockIdx.x] = out_len;

  out_len = ((out_len + 4 - 1) / 4) * 4;

  __syncwarp();
  for (int offset = 16; offset > 0; offset /= 2)
    out_len += __shfl_down_sync(FULL_MASK, out_len, offset);

  __syncwarp();
  if (threadIdx.x == 0) {
    out_len = ((out_len + 128 - 1) / 128) * 128;

    blk_offset_ptr[blockIdx.x + 1] = out_len;
  }
}

template <typename READ_COL_TYPE, typename DATA_TYPE, size_t in_buff_len = 4>
__device__ void
compression_init_warp_orig(queue<DATA_TYPE> &rq, uint64_t *col_len_ptr,
                           uint64_t *blk_offset_ptr, uint64_t CHUNK_SIZE,
                           int COMP_COL_LEN) {

  uint32_t read_bytes = sizeof(DATA_TYPE);

  uint64_t used_bytes = 0;
  uint64_t my_chunk_size = CHUNK_SIZE;

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

  bool reset_flag = false;

  // first data
  DATA_TYPE read_data = 0;
  rq.dequeue(&read_data);
  delta_count = 1;
  prev_val = read_data;

  data_buffer[data_buffer_head] = read_data;
  data_buffer_head = (data_buffer_head + 1) % 2;
  data_buffer_count++;
  used_bytes += read_bytes;

  rq.dequeue(&read_data);

  // second data
  int64_t temp_diff = read_data - prev_val;

  if (temp_diff > 127 || temp_diff < -128) {
    delta_flag = false;
    lit_idx = out_offset;

    // enqueue_header_holder(wq, 1);
    out_len += 1;
    lit_count = 1;

    DATA_TYPE lit_val = data_buffer[0];

    out_len += get_varint_size<DATA_TYPE>(lit_val);

    data_buffer_count--;
  }

  else {
    delta_flag = true;
    cur_delta = (int8_t)temp_diff;
    // if(threadIdx.x == 17 && blockIdx.x == 7 ) printf("cur delta: %x temp_diff
    // %lx  \n",  cur_delta, temp_diff);

    delta_count++;
  }

  prev_val = read_data;
  data_buffer[data_buffer_head] = read_data;
  data_buffer_head = (data_buffer_head + 1) % 2;
  data_buffer_count++;
  used_bytes += read_bytes;

  while (used_bytes < my_chunk_size) {
    // if(threadIdx.x == 24 && blockIdx.x == 93064 ) printf("out len: %llu\n",
    // out_len);

    if (lit_count == 127) {
      lit_count = 0;
      // reset_flag = true;
      // delta_count = 1;
    }

    rq.dequeue(&read_data);
    used_bytes += read_bytes;

    if (delta_count == 1) {

      temp_diff = read_data - prev_val;
      if (temp_diff > 127 || temp_diff < -128) {
        delta_flag = false;
        if (lit_count == 0) {
          lit_idx = out_offset;
          // enqueue_header_holder(wq, 1);
          out_len += 1;
        }

        int8_t data_buffer_tail = (data_buffer_head == 0) ? 1 : 0;
        DATA_TYPE lit_val = data_buffer[data_buffer_tail];
        out_len += get_varint_size<DATA_TYPE>(lit_val);

        lit_count++;
        data_buffer_count--;
      } else {
        delta_flag = true;
        cur_delta = (int8_t)temp_diff;
        // if(threadIdx.x == 17 && blockIdx.x == 7 ) printf("cur delta3: %x
        // temp_diff %lx  \n",  cur_delta, temp_diff);

        delta_count++;
      }
      prev_val = read_data;
      data_buffer[data_buffer_head] = read_data;
      data_buffer_head = (data_buffer_head + 1) % 2;
      data_buffer_count++;

      continue;
    }

    // matched
    if (prev_val + cur_delta == read_data && delta_flag) {
      delta_count++;
      if (delta_count == 3) {

        delta_first_val = data_buffer[data_buffer_head];

        if (lit_count != 0) {
          lit_count = 0;
        }
      }
      // max
      else if (delta_count == 130) {
        int8_t write_byte = delta_count - 4;
        // enqueue_byte(wq, write_byte);
        write_byte = (uint8_t)cur_delta;

        // enqueue_byte(wq, write_byte);

        out_len += 2;
        out_len += get_varint_size(delta_first_val);

        delta_count = 1;
        data_buffer_count = 0;
        lit_count = 0;
      }

      data_buffer[data_buffer_head] = read_data;
      data_buffer_head = (data_buffer_head + 1) % 2;
      data_buffer_count = min(data_buffer_count + 1, 2);
      prev_val = read_data;
    }

    // not matched

    else {
      if (delta_count >= 3) {

        int8_t write_byte = delta_count - 3;

        // enqueue_byte(wq, write_byte);

        write_byte = (uint8_t)cur_delta;

        // enqueue_byte(wq, write_byte);

        out_len += 2;
        out_len += get_varint_size<DATA_TYPE>(delta_first_val);

        // if(threadIdx.x == 17 && blockIdx.x == 7 ) printf("write_byte: %x cur
        // delta:%x  fv: %x\n", delta_count - 3, cur_delta, delta_first_val);

        delta_count = 1;
        data_buffer_count = 0;
        lit_count = 0;
        data_buffer[data_buffer_head] = read_data;
        data_buffer_head = (data_buffer_head + 1) % 2;
        data_buffer_count = min(data_buffer_count + 1, 2);
        prev_val = read_data;
      }

      else {
        if (lit_count == 0) {
          lit_idx = out_offset;
          // write_byte_op(out_buffer_ptr, &out_bytes, (uint8_t) 3, &out_offset,
          // s_col_len, &col_counter, COMP_WRITE_BYTES);
          // enqueue_header_holder(wq, 1);
          out_len++;
        }
        lit_count++;
        DATA_TYPE lit_val = data_buffer[data_buffer_head];

        // write_varint(wq, lit_val);
        out_len += get_varint_size<DATA_TYPE>(lit_val);

        if (lit_count == 127) {
          // out_buffer_ptr[lit_idx] = static_cast<int8_t>(-lit_count);
          //  if(threadIdx.x == 9 && blockIdx.x == 6) printf("set head2:");

          // enqueue_header_setting(wq, static_cast<int8_t>(-lit_count));

          lit_count = 0;
        }

        int64_t temp_diff = read_data - prev_val;
        if (temp_diff > 127 || temp_diff < -128) {

          if (lit_count == 0) {
            lit_idx = out_offset;
            // enqueue_header_holder(wq,1);
            out_len++;
            // write_byte_op(out_buffer_ptr, &out_bytes, (uint8_t) 3,
            // &out_offset,
            //                          s_col_len, &col_counter,
            //                          COMP_WRITE_BYTES);
          }

          delta_flag = false;
          data_buffer_count = 0;
          int8_t data_buffer_tail = (data_buffer_head == 0) ? 1 : 0;
          DATA_TYPE lit_val = data_buffer[data_buffer_tail];
          // enqueue_varint(wq, lit_val);
          out_len += get_varint_size<DATA_TYPE>(lit_val);

          lit_count++;
          delta_count = 1;
        } else {
          data_buffer_count = 1;
          delta_flag = true;
          cur_delta = (int8_t)temp_diff;
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
    // enqueue_byte(wq, write_byte);

    write_byte = (uint8_t)cur_delta;
    // enqueue_byte(wq, write_byte);
    out_len += 2;
    out_len += get_varint_size<DATA_TYPE>(delta_first_val);
    // if(threadIdx.x == 24 && blockIdx.x == 93064 ) printf("out len1: %llu\n",
    // out_len);

  }

  else {
    // update lit count

    if (data_buffer_count == 1) {
      if (lit_count == 127)
        lit_count = 0;

      if (lit_count == 0)
        out_len++;

      int8_t data_buffer_tail = (data_buffer_head == 0) ? 1 : 0;
      DATA_TYPE lit_val = data_buffer[data_buffer_tail];
      out_len += get_varint_size<DATA_TYPE>(lit_val);
      lit_count++;
    }

    if (data_buffer_count == 2) {

      if (lit_count == 127)
        lit_count = 0;

      if (lit_count == 0)
        out_len++;
      DATA_TYPE lit_val = data_buffer[data_buffer_head];
      // enqueue_varint(wq, lit_val);
      out_len += get_varint_size<DATA_TYPE>(lit_val);

      lit_count++;

      if (lit_count == 127)
        lit_count = 0;

      if (lit_count == 0)
        out_len++;

      data_buffer_head = (data_buffer_head + 1) % 2;
      lit_val = data_buffer[data_buffer_head];
      // enqueue_varint(wq, lit_val);
      out_len += get_varint_size<DATA_TYPE>(lit_val);
      lit_count++;
    }
  }

  // col_len_ptr[threadIdx.x + 32 * blockIdx.x] = out_len;

  // out_len = ((out_len + 4 - 1) / 4) * 4;

  // __syncwarp();
  // for (int offset = 16; offset > 0; offset /= 2)
  //     out_len += __shfl_down_sync(FULL_MASK, out_len, offset);

  __syncwarp();
  // out_len = ((out_len + 128 - 1) / 128) * 128;
  // blk_offset_ptr[blockIdx.x + 1] = out_len;
  col_len_ptr[threadIdx.x + 32 * blockIdx.x] = out_len;
  out_len = ((out_len + 4 - 1) / 4) * 4;

  blk_offset_ptr[blockIdx.x * 32 + threadIdx.x + 1] = out_len;
}

__device__ void write_byte_op(int8_t *out_buffer, uint64_t *out_bytes_ptr,
                              uint8_t write_byte, uint64_t *out_offset_ptr,
                              uint64_t *col_len, int COMP_WRITE_BYTES) {
  // if(blockIdx.x == 1 && threadIdx.x == 0) printf("val: %u\n", write_byte);

  out_buffer[*out_offset_ptr] = write_byte;
  (*out_bytes_ptr) = (*out_bytes_ptr) + 1;

  // update the offset
  uint64_t out_offset = (*out_offset_ptr);

  if (((*out_bytes_ptr)) % (COMP_WRITE_BYTES) != 0) {
    *out_offset_ptr = out_offset + 1;
  }

  else {

    for (int i = 0; i < 32; i++) {
      if (col_len[i] > 0 && i != threadIdx.x) {
        out_offset += COMP_WRITE_BYTES;
        col_len[i] -= COMP_WRITE_BYTES;
      }
    }
    *out_offset_ptr = out_offset + 1;
  }
}

__device__ void write_byte_op_orig(int8_t *out_buffer, uint8_t write_byte,
                                   uint64_t *out_offset_ptr) {
  // if(blockIdx.x == 1 && threadIdx.x == 0) printf("val: %u\n", write_byte);

  out_buffer[*out_offset_ptr] = write_byte;
  *out_offset_ptr = (*out_offset_ptr) + 1;
}

template <typename INPUT_T>
__device__ void write_varint_op(int8_t *out_buffer, uint64_t *out_bytes_ptr,
                                INPUT_T val, uint64_t *out_offset_ptr,
                                uint64_t *col_len, int COMP_WRITE_BYTES) {
  INPUT_T write_val = val;
  int8_t write_byte = 0;
  do {
    write_byte = write_val & 0x7F;
    if ((write_val & (~0x7f)) != 0)
      write_byte = write_byte | 0x80;

    // write byte
    write_byte_op(out_buffer, out_bytes_ptr, write_byte, out_offset_ptr,
                  col_len, COMP_WRITE_BYTES);
    write_val = write_val >> 7;
  } while (write_val != 0);
}

template <typename INPUT_T>
__device__ void write_varint_op_orig(int8_t *out_buffer, INPUT_T val,
                                     uint64_t *out_offset_ptr) {
  INPUT_T write_val = val;
  int8_t write_byte = 0;
  do {
    write_byte = write_val & 0x7F;
    if ((write_val & (~0x7f)) != 0)
      write_byte = write_byte | 0x80;

    // write byte
    write_byte_op_orig(out_buffer, write_byte, out_offset_ptr);
    write_val = write_val >> 7;
  } while (write_val != 0);
}

template <typename READ_COL_TYPE, typename DATA_TYPE, typename DECOMP_TYPE,
          size_t in_buff_len = 4>
__device__ void compression_warp(int8_t *out_buffer, queue<DATA_TYPE> &rq,
                                 queue<write_queue_ele> &wq,
                                 uint64_t *col_len_ptr,
                                 uint64_t *blk_offset_ptr, uint64_t CHUNK_SIZE,
                                 int COMP_COL_LEN) {

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

  uint64_t out_offset = threadIdx.x * sizeof(DECOMP_TYPE);
  int COMP_WRITE_BYTES = sizeof(DECOMP_TYPE);

  uint64_t out_bytes = 0;
  uint64_t out_start_idx = blk_offset_ptr[blockIdx.x];
  int8_t *out_buffer_ptr = &(out_buffer[out_start_idx]);

  bool reset_flag = false;

  uint64_t s_col_len[32];
  for (int i = 0; i < 32; i++) {
    s_col_len[i] = ((col_len_ptr[blockIdx.x * 32 + i] + COMP_WRITE_BYTES - 1) /
                    COMP_WRITE_BYTES) *
                   COMP_WRITE_BYTES;
  }
  for (int i = 0; i < threadIdx.x; i++) {
    s_col_len[i] -= COMP_WRITE_BYTES;
  }

  // first data
  DATA_TYPE read_data = 0;
  rq.dequeue(&read_data);
  // if(threadIdx.x == 9 && blockIdx.x == 6) printf("readdata: %x\n",
  // read_data);

  delta_count = 1;
  prev_val = read_data;

  data_buffer[data_buffer_head] = read_data;
  data_buffer_head = (data_buffer_head + 1) % 2;
  data_buffer_count++;
  used_bytes += read_bytes;

  rq.dequeue(&read_data);
  // f(threadIdx.x == 9 && blockIdx.x == 6) printf("readdata: %x\n", read_data);

  // second data
  int64_t temp_diff = read_data - prev_val;

  if (temp_diff > 127 || temp_diff < -128) {
    delta_flag = false;
    lit_idx = out_offset;

    // enqueue_header_holder(wq, 1);

    write_varint_op<DATA_TYPE>(out_buffer_ptr, &out_bytes, 1, &out_offset,
                               s_col_len, COMP_WRITE_BYTES);

    lit_count = 1;

    DATA_TYPE lit_val = data_buffer[0];
    // enqueue_varint<DATA_TYPE>(wq, lit_val);
    write_varint_op<DATA_TYPE>(out_buffer_ptr, &out_bytes, lit_val, &out_offset,
                               s_col_len, COMP_WRITE_BYTES);

    data_buffer_count--;
  }

  else {
    delta_flag = true;
    cur_delta = (int8_t)temp_diff;
    // if(threadIdx.x == 17 && blockIdx.x == 7 ) printf("cur delta: %x temp_diff
    // %lx  \n",  cur_delta, temp_diff);

    delta_count++;
  }

  prev_val = read_data;
  data_buffer[data_buffer_head] = read_data;
  data_buffer_head = (data_buffer_head + 1) % 2;
  data_buffer_count++;
  used_bytes += read_bytes;

  while (used_bytes < my_chunk_size) {

    if (lit_count == 127) {
      // out_buffer_ptr[lit_idx] = static_cast<int8_t>(-lit_count);
      //   enqueue_header_setting(wq, static_cast<int8_t>(-lit_count) );
      out_buffer_ptr[lit_idx] = static_cast<int8_t>(-lit_count);

      // printf("ub: %llu\n", used_bytes);
      lit_count = 0;
      // reset_flag = true;
      // delta_count = 1;
    }

    rq.dequeue(&read_data);
    //    if(threadIdx.x == 9 && blockIdx.x == 6) printf("readdata: %x\n",
    //    read_data);

    // if(threadIdx.x == 11 && blockIdx.x == 0){
    //     printf("data: %x\n", read_data);
    // }

    // uint64_t test_off = (used_bytes / COMP_COL_LEN) * (COMP_COL_LEN * 32) +
    // threadIdx.x * COMP_COL_LEN + blockIdx.x * CHUNK_SIZE + (used_bytes %
    // COMP_COL_LEN);

    // if(test_off >= 72351570  &&  test_off <= 72351597 ){
    //     printf("bid: %i tid:%i offset: %llu read data: %x\n", blockIdx.x,
    //     test_off, read_data);
    // }

    used_bytes += read_bytes;

    //   if(reset_flag){
    //     prev_val = read_data;
    //     data_buffer[data_buffer_head] = read_data;
    //     data_buffer_head = (data_buffer_head + 1) % 2;
    //     data_buffer_count++;

    //     reset_flag = false;
    //     continue;
    // }

    if (delta_count == 1) {

      temp_diff = read_data - prev_val;
      if (temp_diff > 127 || temp_diff < -128) {
        delta_flag = false;
        if (lit_count == 0) {
          lit_idx = out_offset;
          // enqueue_header_holder(wq, 1);
          write_varint_op<DATA_TYPE>(out_buffer_ptr, &out_bytes, 1, &out_offset,
                                     s_col_len, COMP_WRITE_BYTES);
        }

        int8_t data_buffer_tail = (data_buffer_head == 0) ? 1 : 0;
        DATA_TYPE lit_val = data_buffer[data_buffer_tail];
        // enqueue_varint<DATA_TYPE>(wq, lit_val);
        write_varint_op<DATA_TYPE>(out_buffer_ptr, &out_bytes, lit_val,
                                   &out_offset, s_col_len, COMP_WRITE_BYTES);
        lit_count++;
        data_buffer_count--;
      } else {
        delta_flag = true;
        cur_delta = (int8_t)temp_diff;
        // if(threadIdx.x == 17 && blockIdx.x == 7 ) printf("cur delta3: %x
        // temp_diff %lx  \n",  cur_delta, temp_diff);

        delta_count++;
      }
      prev_val = read_data;
      data_buffer[data_buffer_head] = read_data;
      data_buffer_head = (data_buffer_head + 1) % 2;
      data_buffer_count++;

      continue;
    }

    // matched
    if (prev_val + cur_delta == read_data && delta_flag) {
      delta_count++;
      if (delta_count == 3) {
        // if(threadIdx.x == 17 && blockIdx.x == 7 ) printf("cur delta2: %x
        // temp_diff %lx  \n",  cur_delta, temp_diff);

        delta_first_val = data_buffer[data_buffer_head];

        if (lit_count != 0) {
          // if(threadIdx.x == 9 && blockIdx.x == 6) printf("set head1:");

          // enqueue_header_setting(wq, static_cast<int8_t>(-lit_count));
          out_buffer_ptr[lit_idx] = static_cast<int8_t>(-lit_count);

          lit_count = 0;
        }
      }
      // max
      else if (delta_count == 130) {
        int8_t write_byte = delta_count - 4;
        // enqueue_byte(wq, write_byte);
        write_byte_op(out_buffer_ptr, &out_bytes, write_byte, &out_offset,
                      s_col_len, COMP_WRITE_BYTES);

        write_byte = (uint8_t)cur_delta;

        // enqueue_byte(wq, write_byte);
        write_byte_op(out_buffer_ptr, &out_bytes, write_byte, &out_offset,
                      s_col_len, COMP_WRITE_BYTES);
        // enqueue_varint<DATA_TYPE>(wq, delta_first_val);

        write_varint_op<DATA_TYPE>(out_buffer_ptr, &out_bytes, delta_first_val,
                                   &out_offset, s_col_len, COMP_WRITE_BYTES);
        delta_count = 1;
        data_buffer_count = 0;
        lit_count = 0;
      }

      data_buffer[data_buffer_head] = read_data;
      data_buffer_head = (data_buffer_head + 1) % 2;
      data_buffer_count = min(data_buffer_count + 1, 2);
      prev_val = read_data;
    }

    // not matched

    else {
      if (delta_count >= 3) {

        int8_t write_byte = delta_count - 3;

        // enqueue_byte(wq, write_byte);
        write_byte_op(out_buffer_ptr, &out_bytes, write_byte, &out_offset,
                      s_col_len, COMP_WRITE_BYTES);

        write_byte = (uint8_t)cur_delta;

        // enqueue_byte(wq, write_byte);
        write_byte_op(out_buffer_ptr, &out_bytes, write_byte, &out_offset,
                      s_col_len, COMP_WRITE_BYTES);
        // enqueue_varint<DATA_TYPE>(wq, delta_first_val);
        write_varint_op<DATA_TYPE>(out_buffer_ptr, &out_bytes, delta_first_val,
                                   &out_offset, s_col_len, COMP_WRITE_BYTES);

        // if(threadIdx.x == 17 && blockIdx.x == 7 ) printf("write_byte: %x cur
        // delta:%x  fv: %x\n", delta_count - 3, cur_delta, delta_first_val);

        delta_count = 1;
        data_buffer_count = 0;
        lit_count = 0;
        data_buffer[data_buffer_head] = read_data;
        data_buffer_head = (data_buffer_head + 1) % 2;
        data_buffer_count = min(data_buffer_count + 1, 2);
        prev_val = read_data;
      }

      else {
        if (lit_count == 0) {
          lit_idx = out_offset;
          // write_byte_op(out_buffer_ptr, &out_bytes, (uint8_t) 3, &out_offset,
          // s_col_len, &col_counter, COMP_WRITE_BYTES);
          // enqueue_header_holder(wq, 1);
          write_byte_op(out_buffer_ptr, &out_bytes, 3, &out_offset, s_col_len,
                        COMP_WRITE_BYTES);
        }
        lit_count++;
        DATA_TYPE lit_val = data_buffer[data_buffer_head];

        // write_varint(wq, lit_val);
        // enqueue_varint<DATA_TYPE>(wq, lit_val);
        write_varint_op<DATA_TYPE>(out_buffer_ptr, &out_bytes, lit_val,
                                   &out_offset, s_col_len, COMP_WRITE_BYTES);
        if (lit_count == 127) {
          // out_buffer_ptr[lit_idx] = static_cast<int8_t>(-lit_count);
          //  if(threadIdx.x == 9 && blockIdx.x == 6) printf("set head2:");

          // enqueue_header_setting(wq, static_cast<int8_t>(-lit_count));
          out_buffer_ptr[lit_idx] = static_cast<int8_t>(-lit_count);

          lit_count = 0;
        }

        int64_t temp_diff = read_data - prev_val;
        if (temp_diff > 127 || temp_diff < -128) {

          if (lit_count == 0) {
            lit_idx = out_offset;
            // enqueue_header_holder(wq,1);
            write_byte_op(out_buffer_ptr, &out_bytes, (uint8_t)3, &out_offset,
                          s_col_len, COMP_WRITE_BYTES);
          }

          delta_flag = false;
          data_buffer_count = 0;
          int8_t data_buffer_tail = (data_buffer_head == 0) ? 1 : 0;
          DATA_TYPE lit_val = data_buffer[data_buffer_tail];
          // enqueue_varint<DATA_TYPE>(wq, lit_val);
          write_varint_op<DATA_TYPE>(out_buffer_ptr, &out_bytes, lit_val,
                                     &out_offset, s_col_len, COMP_WRITE_BYTES);
          lit_count++;
          delta_count = 1;
        } else {
          data_buffer_count = 1;
          delta_flag = true;
          cur_delta = (int8_t)temp_diff;
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
    // enqueue_byte(wq, write_byte);
    write_byte_op(out_buffer_ptr, &out_bytes, write_byte, &out_offset,
                  s_col_len, COMP_WRITE_BYTES);

    write_byte = (uint8_t)cur_delta;
    // enqueue_byte(wq, write_byte);
    write_byte_op(out_buffer_ptr, &out_bytes, write_byte, &out_offset,
                  s_col_len, COMP_WRITE_BYTES);
    // enqueue_varint<DATA_TYPE>(wq, delta_first_val);
    write_varint_op<DATA_TYPE>(out_buffer_ptr, &out_bytes, delta_first_val,
                               &out_offset, s_col_len, COMP_WRITE_BYTES);
  }

  else {
    // update lit count

    if (data_buffer_count == 1) {
      if (lit_count == 127) {
        // enqueue_header_setting(wq, static_cast<int8_t>(-lit_count));
        out_buffer_ptr[lit_idx] = static_cast<int8_t>(-lit_count);

        lit_count = 0;
      }

      if (lit_count == 0) {
        // enqueue_header_holder(wq, 1);
        lit_idx = out_offset;
        write_varint_op<DATA_TYPE>(out_buffer_ptr, &out_bytes, 1, &out_offset,
                                   s_col_len, COMP_WRITE_BYTES);
      }

      int8_t data_buffer_tail = (data_buffer_head == 0) ? 1 : 0;
      DATA_TYPE lit_val = data_buffer[data_buffer_tail];
      // enqueue_varint(wq, lit_val);
      write_varint_op<DATA_TYPE>(out_buffer_ptr, &out_bytes, lit_val,
                                 &out_offset, s_col_len, COMP_WRITE_BYTES);
      lit_count++;
    }

    if (data_buffer_count == 2) {
      if (lit_count == 127) {
        // enqueue_header_setting(wq, (-lit_count));
        out_buffer_ptr[lit_idx] = static_cast<int8_t>(-lit_count);

        lit_count = 0;
      }

      if (lit_count == 0) {
        // enqueue_header_holder(wq, 1);
        lit_idx = out_offset;
        write_byte_op(out_buffer_ptr, &out_bytes, 1, &out_offset, s_col_len,
                      COMP_WRITE_BYTES);
      }

      DATA_TYPE lit_val = data_buffer[data_buffer_head];
      // enqueue_varint<DATA_TYPE>(wq, lit_val);
      write_varint_op<DATA_TYPE>(out_buffer_ptr, &out_bytes, lit_val,
                                 &out_offset, s_col_len, COMP_WRITE_BYTES);

      lit_count++;

      if (lit_count == 127) {
        // enqueue_header_setting(wq, static_cast<int8_t>(-lit_count));
        out_buffer_ptr[lit_idx] = static_cast<int8_t>(-lit_count);

        lit_count = 0;
      }

      if (lit_count == 0) {
        // enqueue_header_holder(wq, 1);
        lit_idx = out_offset;
        write_byte_op(out_buffer_ptr, &out_bytes, 1, &out_offset, s_col_len,
                      COMP_WRITE_BYTES);
      }

      data_buffer_head = (data_buffer_head + 1) % 2;
      lit_val = data_buffer[data_buffer_head];
      // enqueue_varint<DATA_TYPE>(wq, lit_val);
      write_varint_op<DATA_TYPE>(out_buffer_ptr, &out_bytes, lit_val,
                                 &out_offset, s_col_len, COMP_WRITE_BYTES);

      lit_count++;
    }
    //  if(threadIdx.x == 9 && blockIdx.x == 6) printf("set head3:");

    // enqueue_header_setting(wq, static_cast<int8_t>(-lit_count));
    out_buffer_ptr[lit_idx] = (-lit_count);
  }

  // write_queue_ele done_data;
  // done_data.data = 0;

  // done_data.done = true;
  // wq.enqueue(&done_data);
}

template <typename DECOMP_COL_TYPE, typename DATA_TYPE>
__device__ void compression_writer_warp(queue<write_queue_ele> &wq,
                                        compress_output<DECOMP_COL_TYPE> &out,
                                        uint64_t CHUNK_SIZE) {

  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("size of decomp type %i\n", sizeof(DECOMP_COL_TYPE));
  uint32_t done = 0;
  bool cur_done = false;
  DECOMP_COL_TYPE out_data = 0;
  uint64_t header_offset = 0;
  uint64_t offset_in_word = 0;
  int counter = 0;
  bool header_flag = false;
  int header_word_idx = 0;
  uint64_t iter = 0;

  while (!done) {
    out_data = 0;
    for (int i = 0; i < sizeof(DECOMP_COL_TYPE);) {

      write_queue_ele deq_data;

      if (!cur_done) {
        wq.dequeue(&deq_data);

        if (deq_data.done == true) {
          out.set_flag();
          cur_done = true;
        }

        else {

          if (deq_data.header == 1) {
            // header_offset =  out.get_offset();
            offset_in_word = i;
            header_flag = true;
            header_word_idx = iter;
            deq_data.data = 0;
          }

          if (deq_data.header == 2) {
            // if(header_word_idx == 0) printf("tid: %i bid: %i head:
            // %x\n",threadIdx.x, blockIdx.x,  deq_data.data);

            if (iter == header_word_idx) {

              DECOMP_COL_TYPE cur_data = (deq_data.data & 0x00FF);
              out_data = out_data | (cur_data << (offset_in_word * 8));

            } else {

              out.update_header(header_offset, offset_in_word,
                                (uint8_t)(deq_data.data & 0x00FF));
              header_flag = false;
            }
            continue;
          }

          // if(threadIdx.x == 0 && blockIdx.x == 0){
          //     printf("data: %x\n", deq_data.data);
          // }

          DECOMP_COL_TYPE cur_data = (deq_data.data & 0x00FF);

          // if(threadIdx.x == 17 && blockIdx.x == 7 ) printf("data: %x\n",
          // cur_data);

          counter++;
          out_data = out_data | (cur_data << (i * 8));
          // if(threadIdx.x == 0 && blockIdx.x == 0 && counter < 20) printf("out
          // i:%i offset: %i data: %x\n", i, (i * 8), out_data);
        }
      }
      i++;
    }

    uint64_t cur_offset = out.warp_write(out_data);
    if (header_flag && header_word_idx == iter)
      header_offset = cur_offset;
    iter++;

    done = __ballot_sync(FULL_MASK, !cur_done) == 0;
  }
}

template <typename READ_COL_TYPE, typename DATA_TYPE, typename DECOMP_TYPE,
          size_t in_buff_len = 4>
__device__ void compression_warp_orig(int8_t *out_buffer, queue<DATA_TYPE> &rq,
                                      queue<write_queue_ele> &wq,
                                      uint64_t *col_len_ptr,
                                      uint64_t *blk_offset_ptr,
                                      uint64_t CHUNK_SIZE, int COMP_COL_LEN) {

  uint32_t read_bytes = sizeof(DATA_TYPE);

  uint64_t used_bytes = 0;
  uint64_t my_chunk_size = CHUNK_SIZE;

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

  uint64_t out_offset = 0;

  int COMP_WRITE_BYTES = sizeof(DECOMP_TYPE);

  uint64_t out_bytes = 0;
  uint64_t out_start_idx = blk_offset_ptr[blockIdx.x * 32 + threadIdx.x];
  int8_t *out_buffer_ptr = &(out_buffer[out_start_idx]);

  bool reset_flag = false;

  uint64_t s_col_len[32];
  for (int i = 0; i < 32; i++) {
    s_col_len[i] = ((col_len_ptr[blockIdx.x * 32 + i] + COMP_WRITE_BYTES - 1) /
                    COMP_WRITE_BYTES) *
                   COMP_WRITE_BYTES;
  }
  for (int i = 0; i < threadIdx.x; i++) {
    s_col_len[i] -= COMP_WRITE_BYTES;
  }

  // first data
  DATA_TYPE read_data = 0;
  rq.dequeue(&read_data);
  // if(threadIdx.x == 9 && blockIdx.x == 6) printf("readdata: %x\n",
  // read_data);

  delta_count = 1;
  prev_val = read_data;

  data_buffer[data_buffer_head] = read_data;
  data_buffer_head = (data_buffer_head + 1) % 2;
  data_buffer_count++;
  used_bytes += read_bytes;

  rq.dequeue(&read_data);
  // f(threadIdx.x == 9 && blockIdx.x == 6) printf("readdata: %x\n", read_data);

  // second data
  int64_t temp_diff = read_data - prev_val;

  if (temp_diff > 127 || temp_diff < -128) {
    delta_flag = false;
    lit_idx = out_offset;

    write_varint_op_orig<DATA_TYPE>(out_buffer_ptr, 1, &out_offset);

    lit_count = 1;

    DATA_TYPE lit_val = data_buffer[0];
    // enqueue_varint<DATA_TYPE>(wq, lit_val);
    write_varint_op_orig<DATA_TYPE>(out_buffer_ptr, lit_val, &out_offset);

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

  while (used_bytes < my_chunk_size) {

    if (lit_count == 127) {

      out_buffer_ptr[lit_idx] = static_cast<int8_t>(-lit_count);
      lit_count = 0;
    }

    rq.dequeue(&read_data);

    used_bytes += read_bytes;

    if (delta_count == 1) {

      temp_diff = read_data - prev_val;
      if (temp_diff > 127 || temp_diff < -128) {
        delta_flag = false;
        if (lit_count == 0) {
          lit_idx = out_offset;
          // enqueue_header_holder(wq, 1);
          write_varint_op_orig<DATA_TYPE>(out_buffer_ptr, 1, &out_offset);
        }

        int8_t data_buffer_tail = (data_buffer_head == 0) ? 1 : 0;
        DATA_TYPE lit_val = data_buffer[data_buffer_tail];
        // enqueue_varint<DATA_TYPE>(wq, lit_val);
        write_varint_op_orig<DATA_TYPE>(out_buffer_ptr, lit_val, &out_offset);
        lit_count++;
        data_buffer_count--;
      } else {
        delta_flag = true;
        cur_delta = (int8_t)temp_diff;
        // if(threadIdx.x == 17 && blockIdx.x == 7 ) printf("cur delta3: %x
        // temp_diff %lx  \n",  cur_delta, temp_diff);

        delta_count++;
      }
      prev_val = read_data;
      data_buffer[data_buffer_head] = read_data;
      data_buffer_head = (data_buffer_head + 1) % 2;
      data_buffer_count++;

      continue;
    }

    // matched
    if (prev_val + cur_delta == read_data && delta_flag) {
      delta_count++;
      if (delta_count == 3) {
        // if(threadIdx.x == 17 && blockIdx.x == 7 ) printf("cur delta2: %x
        // temp_diff %lx  \n",  cur_delta, temp_diff);

        delta_first_val = data_buffer[data_buffer_head];

        if (lit_count != 0) {
          // if(threadIdx.x == 9 && blockIdx.x == 6) printf("set head1:");

          // enqueue_header_setting(wq, static_cast<int8_t>(-lit_count));
          out_buffer_ptr[lit_idx] = static_cast<int8_t>(-lit_count);

          lit_count = 0;
        }
      }
      // max
      else if (delta_count == 130) {
        int8_t write_byte = delta_count - 4;
        // enqueue_byte(wq, write_byte);
        write_byte_op_orig(out_buffer_ptr, write_byte, &out_offset);

        write_byte = (uint8_t)cur_delta;

        write_byte_op_orig(out_buffer_ptr, write_byte, &out_offset);

        write_varint_op_orig<DATA_TYPE>(out_buffer_ptr, delta_first_val,
                                        &out_offset);
        delta_count = 1;
        data_buffer_count = 0;
        lit_count = 0;
      }

      data_buffer[data_buffer_head] = read_data;
      data_buffer_head = (data_buffer_head + 1) % 2;
      data_buffer_count = min(data_buffer_count + 1, 2);
      prev_val = read_data;
    }

    // not matched

    else {
      if (delta_count >= 3) {

        int8_t write_byte = delta_count - 3;

        // enqueue_byte(wq, write_byte);
        write_byte_op_orig(out_buffer_ptr, write_byte, &out_offset);

        write_byte = (uint8_t)cur_delta;

        write_byte_op_orig(out_buffer_ptr, write_byte, &out_offset);
        write_varint_op_orig<DATA_TYPE>(out_buffer_ptr, delta_first_val,
                                        &out_offset);

        // if(threadIdx.x == 17 && blockIdx.x == 7 ) printf("write_byte: %x cur
        // delta:%x  fv: %x\n", delta_count - 3, cur_delta, delta_first_val);

        delta_count = 1;
        data_buffer_count = 0;
        lit_count = 0;
        data_buffer[data_buffer_head] = read_data;
        data_buffer_head = (data_buffer_head + 1) % 2;
        data_buffer_count = min(data_buffer_count + 1, 2);
        prev_val = read_data;
      }

      else {
        if (lit_count == 0) {
          lit_idx = out_offset;
          // write_byte_op(out_buffer_ptr, &out_bytes, (uint8_t) 3, &out_offset,
          // s_col_len, &col_counter, COMP_WRITE_BYTES);
          // enqueue_header_holder(wq, 1);
          write_byte_op_orig(out_buffer_ptr, 3, &out_offset);
        }
        lit_count++;
        DATA_TYPE lit_val = data_buffer[data_buffer_head];

        // write_varint(wq, lit_val);
        // enqueue_varint<DATA_TYPE>(wq, lit_val);
        write_varint_op_orig<DATA_TYPE>(out_buffer_ptr, lit_val, &out_offset);
        if (lit_count == 127) {
          // out_buffer_ptr[lit_idx] = static_cast<int8_t>(-lit_count);
          //  if(threadIdx.x == 9 && blockIdx.x == 6) printf("set head2:");

          // enqueue_header_setting(wq, static_cast<int8_t>(-lit_count));
          out_buffer_ptr[lit_idx] = static_cast<int8_t>(-lit_count);

          lit_count = 0;
        }

        int64_t temp_diff = read_data - prev_val;
        if (temp_diff > 127 || temp_diff < -128) {

          if (lit_count == 0) {
            lit_idx = out_offset;
            // enqueue_header_holder(wq,1);
            write_byte_op_orig(out_buffer_ptr, (uint8_t)3, &out_offset);
          }

          delta_flag = false;
          data_buffer_count = 0;
          int8_t data_buffer_tail = (data_buffer_head == 0) ? 1 : 0;
          DATA_TYPE lit_val = data_buffer[data_buffer_tail];
          // enqueue_varint<DATA_TYPE>(wq, lit_val);
          write_varint_op_orig<DATA_TYPE>(out_buffer_ptr, lit_val, &out_offset);
          lit_count++;
          delta_count = 1;
        } else {
          data_buffer_count = 1;
          delta_flag = true;
          cur_delta = (int8_t)temp_diff;
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
    // enqueue_byte(wq, write_byte);
    write_byte_op_orig(out_buffer_ptr, write_byte, &out_offset);

    write_byte = (uint8_t)cur_delta;
    // enqueue_byte(wq, write_byte);
    write_byte_op_orig(out_buffer_ptr, write_byte, &out_offset);
    // enqueue_varint<DATA_TYPE>(wq, delta_first_val);
    write_varint_op_orig<DATA_TYPE>(out_buffer_ptr, delta_first_val,
                                    &out_offset);
  }

  else {
    // update lit count

    if (data_buffer_count == 1) {
      if (lit_count == 127) {
        // enqueue_header_setting(wq, static_cast<int8_t>(-lit_count));
        out_buffer_ptr[lit_idx] = static_cast<int8_t>(-lit_count);

        lit_count = 0;
      }

      if (lit_count == 0) {
        // enqueue_header_holder(wq, 1);
        lit_idx = out_offset;
        write_varint_op_orig<DATA_TYPE>(out_buffer_ptr, 1, &out_offset);
      }

      int8_t data_buffer_tail = (data_buffer_head == 0) ? 1 : 0;
      DATA_TYPE lit_val = data_buffer[data_buffer_tail];
      write_varint_op_orig<DATA_TYPE>(out_buffer_ptr, lit_val, &out_offset);
      lit_count++;
    }

    if (data_buffer_count == 2) {
      if (lit_count == 127) {
        out_buffer_ptr[lit_idx] = static_cast<int8_t>(-lit_count);

        lit_count = 0;
      }

      if (lit_count == 0) {
        // enqueue_header_holder(wq, 1);
        lit_idx = out_offset;
        write_byte_op_orig(out_buffer_ptr, 1, &out_offset);
      }

      DATA_TYPE lit_val = data_buffer[data_buffer_head];
      // enqueue_varint<DATA_TYPE>(wq, lit_val);
      write_varint_op_orig<DATA_TYPE>(out_buffer_ptr, lit_val, &out_offset);

      lit_count++;

      if (lit_count == 127) {
        // enqueue_header_setting(wq, static_cast<int8_t>(-lit_count));
        out_buffer_ptr[lit_idx] = static_cast<int8_t>(-lit_count);

        lit_count = 0;
      }

      if (lit_count == 0) {
        // enqueue_header_holder(wq, 1);
        lit_idx = out_offset;
        write_byte_op_orig(out_buffer_ptr, 1, &out_offset);
      }

      data_buffer_head = (data_buffer_head + 1) % 2;
      lit_val = data_buffer[data_buffer_head];
      // enqueue_varint<DATA_TYPE>(wq, lit_val);
      write_varint_op_orig<DATA_TYPE>(out_buffer_ptr, lit_val, &out_offset);

      lit_count++;
    }
    //  if(threadIdx.x == 9 && blockIdx.x == 6) printf("set head3:");

    // enqueue_header_setting(wq, static_cast<int8_t>(-lit_count));
    out_buffer_ptr[lit_idx] = (-lit_count);
  }
}

template <typename READ_COL_TYPE, typename DATA_TYPE, typename OUT_COL_TYPE,
          uint16_t in_queue_size = 4>
__global__ void setup_deflate(uint8_t *input_ptr, uint64_t *col_len_ptr,
                              uint64_t *blk_offset_ptr, uint64_t CHUNK_SIZE,
                              int COMP_COL_LEN) {

  static __shared__ DATA_TYPE in_queue_[32][in_queue_size];
  static __shared__ simt::atomic<uint8_t, simt::thread_scope_block> h[32];
  static __shared__ simt::atomic<uint8_t, simt::thread_scope_block> t[32];

  h[threadIdx.x] = 0;
  t[threadIdx.x] = 0;

  __syncthreads();

  if (threadIdx.y == 0) {
    queue<DATA_TYPE> in_queue(in_queue_[threadIdx.x], h + threadIdx.x,
                              t + threadIdx.x, in_queue_size);
    uint8_t *chunk_ptr = (input_ptr + (CHUNK_SIZE * blockIdx.x));
    compression_reader_warp<READ_COL_TYPE, DATA_TYPE>(
        in_queue, (DATA_TYPE *)chunk_ptr, CHUNK_SIZE, COMP_COL_LEN);
  }

  else if (threadIdx.y == 1) {
    queue<DATA_TYPE> in_queue(in_queue_[threadIdx.x], h + threadIdx.x,
                              t + threadIdx.x, in_queue_size);
    compression_init_warp<READ_COL_TYPE, DATA_TYPE, in_queue_size>(
        in_queue, col_len_ptr, blk_offset_ptr, CHUNK_SIZE, COMP_COL_LEN);
  }
}

template <typename READ_COL_TYPE, typename DATA_TYPE, typename OUT_COL_TYPE,
          uint16_t in_queue_size = 4>
__global__ void setup_deflate_orig(uint8_t *input_ptr, uint64_t *col_len_ptr,
                                   uint64_t *blk_offset_ptr,
                                   uint64_t CHUNK_SIZE, int COMP_COL_LEN) {

  static __shared__ DATA_TYPE in_queue_[32][in_queue_size];
  static __shared__ simt::atomic<uint8_t, simt::thread_scope_block> h[32];
  static __shared__ simt::atomic<uint8_t, simt::thread_scope_block> t[32];

  h[threadIdx.x] = 0;
  t[threadIdx.x] = 0;

  __syncthreads();
  if (threadIdx.y == 0) {
    queue<DATA_TYPE> in_queue(in_queue_[threadIdx.x], h + threadIdx.x,
                              t + threadIdx.x, in_queue_size);
    uint8_t *chunk_ptr = (input_ptr + (CHUNK_SIZE * blockIdx.x * 32));
    compression_reader_warp<READ_COL_TYPE, DATA_TYPE>(
        in_queue, (DATA_TYPE *)chunk_ptr, CHUNK_SIZE * 32, CHUNK_SIZE);
  }

  else if (threadIdx.y == 1) {
    queue<DATA_TYPE> in_queue(in_queue_[threadIdx.x], h + threadIdx.x,
                              t + threadIdx.x, in_queue_size);
    compression_init_warp_orig<READ_COL_TYPE, DATA_TYPE, in_queue_size>(
        in_queue, col_len_ptr, blk_offset_ptr, CHUNK_SIZE, COMP_COL_LEN);
  }
}

template <typename READ_COL_TYPE, typename DATA_TYPE, typename OUT_COL_TYPE,
          uint16_t queue_size = 4>
__global__ void deflate(uint8_t *input_ptr, uint8_t *out, uint64_t *col_len_ptr,
                        uint64_t *blk_offset_ptr, uint64_t CHUNK_SIZE,
                        int COMP_COL_LEN) {

  static __shared__ DATA_TYPE in_queue_[32][queue_size];
  static __shared__ simt::atomic<uint8_t, simt::thread_scope_block> h[32];
  static __shared__ simt::atomic<uint8_t, simt::thread_scope_block> t[32];

  static __shared__ write_queue_ele out_queue_[32][queue_size];
  static __shared__ simt::atomic<uint8_t, simt::thread_scope_block> out_h[32];
  static __shared__ simt::atomic<uint8_t, simt::thread_scope_block> out_t[32];

  h[threadIdx.x] = 0;
  t[threadIdx.x] = 0;
  out_h[threadIdx.x] = 0;
  out_t[threadIdx.x] = 0;

  __syncthreads();

  if (threadIdx.y == 0) {
    queue<DATA_TYPE> in_queue(in_queue_[threadIdx.x], h + threadIdx.x,
                              t + threadIdx.x, queue_size);
    uint8_t *chunk_ptr = (input_ptr + (CHUNK_SIZE * blockIdx.x));
    compression_reader_warp<READ_COL_TYPE, DATA_TYPE>(
        in_queue, (DATA_TYPE *)chunk_ptr, CHUNK_SIZE, COMP_COL_LEN);
  }

  else if (threadIdx.y == 1) {
    queue<DATA_TYPE> in_queue(in_queue_[threadIdx.x], h + threadIdx.x,
                              t + threadIdx.x, queue_size);
    queue<write_queue_ele> out_queue(out_queue_[threadIdx.x],
                                     out_h + threadIdx.x, out_t + threadIdx.x,
                                     queue_size);
    compression_warp<READ_COL_TYPE, DATA_TYPE, OUT_COL_TYPE, queue_size>(
        (int8_t *)out, in_queue, out_queue, col_len_ptr, blk_offset_ptr,
        CHUNK_SIZE, COMP_COL_LEN);
  }
  __syncthreads();
}

template <typename READ_COL_TYPE, typename DATA_TYPE, typename OUT_COL_TYPE,
          uint16_t queue_size = 4>
__global__ void deflate_orig(uint8_t *input_ptr, uint8_t *out,
                             uint64_t *col_len_ptr, uint64_t *blk_offset_ptr,
                             uint64_t CHUNK_SIZE, int COMP_COL_LEN) {

  static __shared__ DATA_TYPE in_queue_[32][queue_size];
  static __shared__ simt::atomic<uint8_t, simt::thread_scope_block> h[32];
  static __shared__ simt::atomic<uint8_t, simt::thread_scope_block> t[32];

  static __shared__ write_queue_ele out_queue_[32][queue_size];
  static __shared__ simt::atomic<uint8_t, simt::thread_scope_block> out_h[32];
  static __shared__ simt::atomic<uint8_t, simt::thread_scope_block> out_t[32];

  h[threadIdx.x] = 0;
  t[threadIdx.x] = 0;
  out_h[threadIdx.x] = 0;
  out_t[threadIdx.x] = 0;

  __syncthreads();

  if (threadIdx.y == 0) {
    queue<DATA_TYPE> in_queue(in_queue_[threadIdx.x], h + threadIdx.x,
                              t + threadIdx.x, queue_size);
    uint8_t *chunk_ptr = (input_ptr + (CHUNK_SIZE * blockIdx.x * 32));
    compression_reader_warp<READ_COL_TYPE, DATA_TYPE>(
        in_queue, (DATA_TYPE *)chunk_ptr, CHUNK_SIZE * 32, CHUNK_SIZE);
  }

  else if (threadIdx.y == 1) {
    queue<DATA_TYPE> in_queue(in_queue_[threadIdx.x], h + threadIdx.x,
                              t + threadIdx.x, queue_size);
    queue<write_queue_ele> out_queue(out_queue_[threadIdx.x],
                                     out_h + threadIdx.x, out_t + threadIdx.x,
                                     queue_size);
    compression_warp_orig<READ_COL_TYPE, DATA_TYPE, OUT_COL_TYPE, queue_size>(
        (int8_t *)out, in_queue, out_queue, col_len_ptr, blk_offset_ptr,
        CHUNK_SIZE, COMP_COL_LEN);
  }
  __syncthreads();
}

__global__ void reduction_scan(uint64_t *blk_offset, uint64_t n) {
  blk_offset[0] = 0;
  for (uint64_t i = 1; i <= n; i++) {
    blk_offset[i] += blk_offset[i - 1];
  }
}
