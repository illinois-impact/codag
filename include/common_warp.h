// #ifndef __ZLIB_H__
// #define __ZLIB_H__

#include <common.h>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <simt/atomic>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

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

#define MAXBITS 15
#define FIXLCODES 288
#define MAXDCODES 30
#define MAXCODES 316

template <typename T> struct volatile_queue {
  T *queue_;
  volatile uint8_t head;
  volatile uint8_t tail;
  uint8_t len;

  __device__ volatile_queue(T *buff, volatile uint8_t h, volatile uint8_t t,
                            const uint8_t l) {
    queue_ = buff;
    head = h;
    tail = t;
    len = l;
  }

  __device__ void enqueue(const T *v) {

    while ((tail + 1) % len == head) {
      __nanosleep(100);
    }
    queue_[tail] = *v;
    tail = (tail + 1) % len;
  }

  __device__ void dequeue(T *v) {

    while (head == tail) {
      __nanosleep(100);
    }
    *v = queue_[head];
    head = (head + 1) % len;
  }
};

template <typename T> struct queue {
  T *queue_;
  simt::atomic<uint8_t, simt::thread_scope_block> *head;
  simt::atomic<uint8_t, simt::thread_scope_block> *tail;
  volatile uint8_t *v_head;
  volatile uint8_t *v_tail;
  volatile T *v_queue_;

  uint8_t len;
  bool vol;

  __device__ queue(T *buff, simt::atomic<uint8_t, simt::thread_scope_block> *h,
                   simt::atomic<uint8_t, simt::thread_scope_block> *t,
                   const uint8_t l, volatile T *v_buff = NULL,
                   volatile uint8_t *v_h = NULL, volatile uint8_t *v_t = NULL,
                   bool v = false) {
    queue_ = buff;
    head = h;
    tail = t;
    len = l;

    v_queue_ = v_buff;
    v_head = v_h;
    v_tail = v_t;
    vol = v;
  }
  __device__ void enqueue(const T *v) {

    const auto cur_tail = tail->load(simt::memory_order_relaxed);
    const auto next_tail = (cur_tail + 1) % len;

    while (next_tail == head->load(simt::memory_order_acquire))
      __nanosleep(50);

    queue_[cur_tail] = *v;
    tail->store(next_tail, simt::memory_order_release);
  }

  __device__ void dequeue(T *v) {

    if (vol) {

      while (*v_head == *v_tail) {
        __nanosleep(100);
      }
      *v = v_queue_[*v_head];
      *v_head = (*v_head + 1) % len;
    }

    else {
      const auto cur_head = head->load(simt::memory_order_relaxed);
      while (cur_head == tail->load(simt::memory_order_acquire))
        __nanosleep(50);

      *v = queue_[cur_head];

      const auto next_head = (cur_head + 1) % len;

      head->store(next_head, simt::memory_order_release);
    }
  }

  __device__ void dequeue_sync(T *v) {

    uint8_t cur_head;
    bool sleep_flag = false;
    if (threadIdx.x == 0)
      cur_head = head->load(simt::memory_order_relaxed);

    while (1) {
      if (threadIdx.x == 0) {
        if (cur_head == tail->load(simt::memory_order_acquire))
          sleep_flag = true;
        else
          sleep_flag = false;
      }
      sleep_flag = __shfl_sync(FULL_MASK, sleep_flag, 0);
      if (sleep_flag)
        __nanosleep(20);
      else
        break;
    }

    if (threadIdx.x == 0) {
      *v = queue_[cur_head];

      const auto next_head = (cur_head + 1) % len;

      head->store(next_head, simt::memory_order_release);
    }
  }

  __device__ void attempt_dequeue(T *v, bool *p) {

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

  __device__ void warp_enqueue(T *v, uint8_t subchunk_idx, uint8_t enq_num) {

    if (vol) {
      T my_v = *v;
      for (uint8_t i = 0; i < enq_num; i++) {
        T cur_v = __shfl_sync(FULL_MASK, my_v, i);
        if (threadIdx.x == subchunk_idx) {

          while ((*v_tail + 1) % len == *v_head) {
            __nanosleep(20);
          }
          // printf("cur_v: %lx\n", cur_v);
          v_queue_[*v_tail] = cur_v;
          *v_tail = ((*v_tail) + 1) % len;
        }
        __syncwarp(FULL_MASK);
      }

    } else {

      T my_v = *v;

      for (uint8_t i = 0; i < enq_num; i++) {
        T cur_v = __shfl_sync(FULL_MASK, my_v, i);
        if (threadIdx.x == subchunk_idx) {

          const auto cur_tail = tail->load(simt::memory_order_relaxed);
          const auto next_tail = (cur_tail + 1) % (len);

          while (next_tail == head->load(simt::memory_order_acquire)) {
            __nanosleep(20);
          }
          queue_[cur_tail] = cur_v;
          tail->store(next_tail, simt::memory_order_release);
        }
        __syncwarp(FULL_MASK);
      }
    }
  }

  __device__ void warp_enqueue_sync(T *v, uint8_t subchunk_idx,
                                    uint8_t enq_num) {
    bool sleep_flag = false;

    T my_v = *v;

    for (uint8_t i = 0; i < enq_num; i++) {
      T cur_v = __shfl_sync(FULL_MASK, my_v, i);
      uint8_t cur_tail;
      uint8_t next_tail;

      if (threadIdx.x == subchunk_idx) {
        cur_tail = tail->load(simt::memory_order_relaxed);
        next_tail = (cur_tail + 1) % (len);
      }

      while (1) {
        if (threadIdx.x == subchunk_idx) {
          if (next_tail == head->load(simt::memory_order_acquire))
            sleep_flag = true;
          else
            sleep_flag = false;
        }
        sleep_flag = __shfl_sync(FULL_MASK, sleep_flag, subchunk_idx);
        if (sleep_flag)
          __nanosleep(200);
        else
          break;
      }

      if (threadIdx.x == subchunk_idx) {

        queue_[cur_tail] = cur_v;
        tail->store(next_tail, simt::memory_order_release);
      }
      __syncwarp(FULL_MASK);
    }
  }

  __device__ void warp_enqueue_multi(T *v, uint8_t subchunk_idx,
                                     uint8_t enq_num) {

    T my_v = *v;
    uint64_t q_ptr = (uint64_t)queue_;
    uint8_t cur_tail = 0;

    if (threadIdx.x == subchunk_idx) {
      cur_tail = tail->load(simt::memory_order_relaxed);
      while (1) {
        const auto cur_head = head->load(simt::memory_order_acquire);
        uint32_t num_ele = cur_tail >= cur_head
                               ? (len - (cur_tail - cur_head) - 1)
                               : (-cur_tail + cur_head - 1);

        if (num_ele >= 32) {
          // printf("idx: %u num ele: %lu len %u head: %lu tail: %lu enq num:
          // %u\n", subchunk_idx, (unsigned long) num_ele, (unsigned) len,
          // (unsigned long) cur_head, (unsigned long) cur_tail, enq_num);

          break;
        }

        __nanosleep(20);
      }
    }

    cur_tail = __shfl_sync(FULL_MASK, cur_tail, subchunk_idx);
    auto next_tail = (cur_tail + enq_num) % len;

    cur_tail = (cur_tail + threadIdx.x) % len;

    T *queue = (T *)(__shfl_sync(FULL_MASK, q_ptr, subchunk_idx));

    if (threadIdx.x < enq_num) {
      // queue_[cur_tail] = my_v;
      queue[cur_tail] = my_v;
    }
    __syncwarp();

    if (threadIdx.x == subchunk_idx) {
      // printf("next tail: %lu\n", next_tail);
      tail->store(next_tail, simt::memory_order_release);
    }
  }

  template <int NUM_THREADS = 16>
  __device__ void sub_warp_enqueue_multi(T *v, uint8_t subchunk_idx,
                                         uint8_t enq_num, uint32_t MASK,
                                         int div) {

    T my_v = *v;
    uint64_t q_ptr = (uint64_t)queue_;
    uint8_t cur_tail = 0;

    int t = threadIdx.x % NUM_THREADS;

    if (threadIdx.x == div * NUM_THREADS) {
      cur_tail = tail->load(simt::memory_order_relaxed);
      while (1) {
        const auto cur_head = head->load(simt::memory_order_acquire);
        uint32_t num_ele = cur_tail >= cur_head
                               ? (len - (cur_tail - cur_head) - 1)
                               : (-cur_tail + cur_head - 1);

        if (num_ele >= NUM_THREADS) {
          break;
        }
        __nanosleep(20);
      }
    }

    cur_tail = __shfl_sync(MASK, cur_tail, div * NUM_THREADS);
    auto next_tail = (cur_tail + enq_num) % len;

    cur_tail = (cur_tail + t) % len;

    T *queue = (T *)(__shfl_sync(MASK, q_ptr, div * NUM_THREADS));

    if (t < enq_num) {
      // queue_[cur_tail] = my_v;
      queue[cur_tail] = my_v;
    }
    __syncwarp(MASK);

    if (threadIdx.x == div * NUM_THREADS) {
      tail->store(next_tail, simt::memory_order_release);
    }
  }

  template <int NUM_THREADS = 16>
  __device__ void sub_warp_enqueue(T *v, uint8_t subchunk_idx, uint8_t enq_num,
                                   uint32_t MASK, int div) {

    T my_v = *v;

    for (uint8_t i = 0; i < enq_num; i++) {
      T cur_v = __shfl_sync(MASK, my_v, i + div * NUM_THREADS);

      if (threadIdx.x == div * NUM_THREADS) {

        const auto cur_tail = tail->load(simt::memory_order_relaxed);
        const auto next_tail = (cur_tail + 1) % (len);

        while (next_tail == head->load(simt::memory_order_acquire)) {
          __nanosleep(20);
        }
        queue_[cur_tail] = cur_v;
        tail->store(next_tail, simt::memory_order_release);
      }
      __syncwarp(MASK);
    }
  }
};

template <typename T, typename QUEUE_TYPE>
__forceinline__ __device__ void comp_enqueue(T *v, queue<QUEUE_TYPE> *q) {
  T temp_v = *v;

  //#pragma unroll
  for (int i = 0; i < (sizeof(T) / sizeof(QUEUE_TYPE)); i++) {

    const auto cur_tail = q->tail->load(simt::memory_order_relaxed);
    const auto next_tail = (cur_tail + 1) % (q->len);

    while (next_tail == q->head->load(simt::memory_order_acquire))
      __nanosleep(20);

    // q->queue_[cur_tail] = (QUEUE_TYPE) (((temp_v >> (i * 8 *
    // sizeof(QUEUE_TYPE))) & (0x0FFFFFFFF))); q->queue_[cur_tail] =
    // (QUEUE_TYPE) (((temp_v >> (i * 8 * sizeof(QUEUE_TYPE))) ));

    if (i == 0)
      q->queue_[cur_tail] = (temp_v).x;
    else if (i == 1)
      q->queue_[cur_tail] = (temp_v).y;
    else if (i == 2)
      q->queue_[cur_tail] = (temp_v).z;
    else if (i == 3)
      q->queue_[cur_tail] = (temp_v).w;

    q->tail->store(next_tail, simt::memory_order_release);
  }
}

template <typename T, typename QUEUE_TYPE>
__forceinline__ __device__ void warp_enqueue(T *v, queue<QUEUE_TYPE> *q,
                                             uint8_t subchunk_idx,
                                             uint8_t enq_num) {
  // printf("cur_v: %lx\n", cur_v);

  T my_v = *v;

  for (uint8_t i = 0; i < enq_num; i++) {
    T cur_v = __shfl_sync(FULL_MASK, my_v, i);
    if (threadIdx.x == subchunk_idx) {

      const auto cur_tail = q->tail->load(simt::memory_order_relaxed);
      const auto next_tail = (cur_tail + 1) % (q->len);

      while (next_tail == q->head->load(simt::memory_order_acquire)) {
        __nanosleep(20);
      }
      // printf("cur_v: %lx\n", cur_v);
      q->queue_[cur_tail] = cur_v;
      q->tail->store(next_tail, simt::memory_order_release);
    }
    __syncwarp(FULL_MASK);
  }
}

// producer of input queue
template <typename COMP_COL_TYPE> struct decompress_input {

  uint32_t row_offset;
  uint32_t len;
  uint32_t read_bytes;
  uint32_t t_read_mask;
  uint64_t pointer_off;

  COMP_COL_TYPE *pointer;

  __device__ decompress_input(const uint8_t *ptr, const uint32_t l,
                              const uint64_t p_off = 0)
      : pointer((COMP_COL_TYPE *)ptr), len(l), pointer_off(p_off) {
    t_read_mask = (0xffffffff >> (32 - threadIdx.x));
    row_offset = 0;
    read_bytes = 0;
  }

  __device__ decompress_input() {}

  __device__ void set_input(const uint8_t *ptr, const uint32_t l,
                            const uint64_t p_off, int num_chunk) {
    if (threadIdx.x < num_chunk) {
      pointer = (COMP_COL_TYPE *)ptr;
      len = l;

      pointer_off = p_off;
      t_read_mask = (0xffffffff >> (32 - threadIdx.x));
      row_offset = 0;
      read_bytes = 0;
    }
  }

  __forceinline__ __device__ int8_t comp_read_data(const uint32_t alivemask,
                                                   COMP_COL_TYPE *v) {

    int8_t read_count = 0;
    bool read = (read_bytes) < len;
    uint32_t read_sync = __ballot_sync(alivemask, read);

    if (__builtin_expect(read_sync == 0, 0)) {
      return -1;
    }

    if (read) {
      *v = pointer[row_offset + __popc(read_sync & t_read_mask)];
      row_offset += __popc(read_sync);
      read_bytes += sizeof(COMP_COL_TYPE);
      read_count = sizeof(COMP_COL_TYPE);
    }

    __syncwarp(alivemask);

    return read_count;
  }
};

template <typename COMP_COL_TYPE>
__forceinline__ __device__ uint8_t
comp_read_data_seq(const uint32_t alivemask, COMP_COL_TYPE *v,
                   decompress_input<COMP_COL_TYPE> &in, uint8_t src_idx) {

  uint32_t src_len =
      __shfl_sync(alivemask, in.len, src_idx) / sizeof(COMP_COL_TYPE);
  uint32_t src_read_bytes = __shfl_sync(alivemask, in.read_bytes, src_idx);
  uint32_t src_pointer_off = __shfl_sync(alivemask, in.pointer_off, src_idx);

  bool read = (src_read_bytes + threadIdx.x) < src_len;
  uint32_t read_sync = __ballot_sync(alivemask, read);

  if (read) {
    *v = in.pointer[src_read_bytes + threadIdx.x + src_pointer_off];
  }

  uint8_t read_count = __popc(read_sync);

  if (threadIdx.x == src_idx) {
    in.read_bytes += read_count;
  }

  return read_count;
}

template <typename COMP_COL_TYPE>
__forceinline__ __device__ uint8_t comp_read_data_seq_shared(
    const uint32_t alivemask, COMP_COL_TYPE *v,
    decompress_input<COMP_COL_TYPE> *in, uint8_t src_idx) {

  uint32_t src_len = in[src_idx].len / sizeof(COMP_COL_TYPE);
  uint32_t src_read_bytes = in[src_idx].read_bytes;
  uint32_t src_pointer_off = in[src_idx].pointer_off;

  bool read = (src_read_bytes + threadIdx.x) < src_len;
  uint32_t read_sync = __ballot_sync(alivemask, read);

  if (read) {
    *v = in[src_idx].pointer[src_read_bytes + threadIdx.x + src_pointer_off];
  }

  uint8_t read_count = __popc(read_sync);

  __syncwarp();
  if (threadIdx.x == src_idx) {
    in[src_idx].read_bytes += read_count;
  }

  return read_count;
}

template <typename COMP_COL_TYPE>
__forceinline__ __device__ uint8_t
comp_read_data_seq_sub(const uint32_t alivemask, COMP_COL_TYPE *v,
                       decompress_input<COMP_COL_TYPE> &in, uint8_t src_idx) {

  uint32_t src_len =
      __shfl_sync(alivemask, in.len, src_idx) / sizeof(COMP_COL_TYPE);
  uint32_t src_read_bytes = __shfl_sync(alivemask, in.read_bytes, src_idx);
  uint64_t src_pointer_off = __shfl_sync(alivemask, in.pointer_off, src_idx);

  bool read = (src_read_bytes + threadIdx.x - src_idx) < src_len;
  uint32_t read_sync = __ballot_sync(alivemask, read);

  if (read) {
    *v = in.pointer[src_read_bytes + threadIdx.x - src_idx + src_pointer_off];
  }

  uint8_t read_count = __popc(read_sync);

  if (threadIdx.x == src_idx) {
    in.read_bytes += read_count;
  }

  return read_count;
}
// consumer of input queue
template <typename READ_COL_TYPE, uint8_t buff_len = 4> struct input_stream {
  uint32_t *buff;
  // union buff{
  //     READ_COL_TYPE* b;
  //     uint32_t* u;
  // }buf;
  uint8_t head;
  uint8_t count;

  queue<READ_COL_TYPE> *q;
  uint32_t read_bytes;
  uint32_t expected_bytes;
  // uint8_t bu_size = (sizeof(READ_COL_TYPE)* buff_len)/sizeof(uint32_t);

  uint8_t uint_bit_offset;

  __device__ input_stream(queue<READ_COL_TYPE> *q_, uint32_t eb,
                          READ_COL_TYPE *shared_b, bool pass = true) {
    // input_stream(queue<READ_COL_TYPE>* q_, uint32_t eb) {
    if (pass) {
      q = q_;
      expected_bytes = eb;

      buff = shared_b;

      head = 0;

      uint_bit_offset = 0;
      read_bytes = 0;
      count = 0;
      for (; (count < buff_len) && (read_bytes < expected_bytes);
           count++, read_bytes += sizeof(uint32_t)) {
        // q->dequeue(b.b + count);
        q->dequeue(buff + count);
      }
    }
  }

  template <typename T>
  __device__ void get_n_bits(const uint32_t n, T *out,
                             bool change_state = true) {

    *out = (T)0;

    uint32_t a_val = buff[(head)];
    uint32_t b_val_idx = (head + 1);
    b_val_idx = b_val_idx % buff_len;
    uint32_t b_val = buff[b_val_idx];

    uint32_t c_val = __funnelshift_rc(a_val, b_val, uint_bit_offset);
    ((uint32_t *)out)[0] = c_val;

    if (32 > n) {
      ((uint32_t *)out)[0] <<= (32 - n);
      ((uint32_t *)out)[0] >>= (32 - n);
    }

    if (change_state) {
      uint_bit_offset += n;
      if (uint_bit_offset >= 32) {
        uint_bit_offset = uint_bit_offset % 32;
        head = (head + 1) % buff_len;

        count--;
      }
    }
  }

  // T should be at least 32bits
  template <typename T> __device__ void fetch_n_bits(const uint32_t n, T *out) {
    while ((count < buff_len) && (read_bytes < expected_bytes)) {
      q->dequeue(buff + ((head + count) % buff_len));

      count++;

      read_bytes += sizeof(uint32_t);
    }

    get_n_bits<T>(n, out);
  }

  template <typename T>
  __device__ void fetch_n_bits_sync(const uint32_t n, T *out) {

    while (1) {
      bool while_flag = true;
      if (threadIdx.x == 0) {
        while_flag = (count < buff_len) && (read_bytes < expected_bytes);
      }
      while_flag = __shfl_sync(FULL_MASK, while_flag, 0);

      if (while_flag) {
        q->dequeue_sync(buff + ((head + count) % buff_len));
        count++;
        read_bytes += sizeof(uint32_t);
      } else
        break;
    }

    if (threadIdx.x == 0)
      get_n_bits<T>(n, out);
  }

  __device__ void skip_n_bits(const uint32_t n) {
    while ((count < buff_len) && (read_bytes < expected_bytes)) {
      q->dequeue(buff + ((head + count) % buff_len));

      count++;

      read_bytes += sizeof(uint32_t);
    }

    uint_bit_offset += n;
    if (uint_bit_offset >= 32) {
      uint_bit_offset = uint_bit_offset % 32;
      head = (head + 1) % buff_len;

      count--;
    }
  }

  __device__ void align_bits() {
    if (uint_bit_offset % 8 != 0) {
      uint_bit_offset = ((uint_bit_offset + 7) / 8) * 8;
      if (uint_bit_offset == 32) {
        uint_bit_offset = 0;
        head = (head + 1) % buff_len;
        count--;
      }
    }
  }

  template <typename T> __device__ void peek_n_bits(const uint32_t n, T *out) {
    while ((count < buff_len) && (read_bytes < expected_bytes)) {
      q->dequeue(buff + ((head + count) % buff_len));
      count++;

      read_bytes += sizeof(uint32_t);
    }

    get_n_bits<T>(n, out, false);
  }

  template <typename T>
  __device__ void peek_n_bits_sync(const uint32_t n, T *out) {

    while (1) {
      bool while_flag = true;
      if (threadIdx.x == 0)
        while_flag = (count < buff_len) && (read_bytes < expected_bytes);
      while_flag = __shfl_sync(FULL_MASK, while_flag, 0);
      if (while_flag) {
        q->dequeue_sync(buff + ((head + count) % buff_len));
        count++;
        read_bytes += sizeof(uint32_t);
      } else
        break;
    }

    if (threadIdx.x == 0)
      get_n_bits<T>(n, out, false);
  }
};

template <typename READ_COL_TYPE, uint8_t buff_len = 4>
struct full_warp_input_stream {

  uint32_t row_offset;
  uint32_t len;
  uint32_t read_bytes_input;
  uint64_t pointer_off;

  uint32_t *in;

  uint8_t head;
  uint16_t count;
  uint8_t uint_bit_offset;

  uint32_t *buff;

  __device__ void fill_buf() {

    while (((count) < 32) && (read_bytes_input < len)) {

      bool read = (read_bytes_input + threadIdx.x) < len;
      if (read) {
        uint8_t head_idx = (head + count + threadIdx.x) % buff_len;
        *(buff + head_idx) = in[read_bytes_input + threadIdx.x + pointer_off];
      }
      uint32_t read_sync = __ballot_sync(FULL_MASK, read);
      uint8_t read_count = __popc(read_sync);

      read_bytes_input += read_count;

      count += 32;
    }
    __syncwarp();
  }

  __device__ full_warp_input_stream(const uint8_t *ptr, const uint32_t l,
                                    const uint64_t p_off, uint32_t *s_buf,
                                    bool fill = false) {
    buff = s_buf;

    head = 0;
    uint_bit_offset = 0;
    count = 0;

    read_bytes_input = 0;
    row_offset = 0;
    pointer_off = p_off;
    len = l;
    in = (uint32_t *)ptr;

    // if(fill) fill_buf();
  }

  template <typename T>
  __device__ void get_n_bits(const uint32_t n, T *out,
                             bool change_state = true) {

    *out = (T)0;

    uint32_t a_val = buff[(head)];

    uint32_t b_val_idx = (head + 1);
    b_val_idx = b_val_idx % buff_len;

    uint32_t b_val = buff[b_val_idx];

    uint32_t c_val = __funnelshift_rc(a_val, b_val, uint_bit_offset);

    ((uint32_t *)out)[0] = c_val;

    if (32 > n) {
      ((uint32_t *)out)[0] <<= (32 - n);
      ((uint32_t *)out)[0] >>= (32 - n);
    }

    if (change_state) {
      uint_bit_offset += n;
      if (uint_bit_offset >= 32) {
        uint_bit_offset = uint_bit_offset % 32;
        head = (head + 1) % buff_len;

        count--;
      }
    }
  }

  template <typename T>
  __device__ void get_n_bits_single(const uint32_t n, T *out,
                                    bool change_state = true) {

    *out = (T)0;

    uint32_t a_val = buff[(head)];

    uint32_t b_val_idx = (head + 1);
    b_val_idx = b_val_idx % buff_len;

    uint32_t b_val = buff[b_val_idx];

    uint32_t c_val = __funnelshift_rc(a_val, b_val, uint_bit_offset);

    ((uint32_t *)out)[0] = c_val;

    if (32 > n) {
      ((uint32_t *)out)[0] <<= (32 - n);
      ((uint32_t *)out)[0] >>= (32 - n);
    }

    if (change_state) {
      uint_bit_offset += n;
      if (uint_bit_offset >= 32) {
        uint_bit_offset = uint_bit_offset % 32;
        head = (head + 1) % buff_len;

        // count--;
      }
    }
    if (change_state)
      count -= n;
  }

  // T should be at least 32bits
  template <typename T> __device__ void fetch_n_bits(const uint32_t n, T *out) {
    fill_buf();
    get_n_bits<T>(n, out);
  }

  template <typename T>
  __device__ bool fetch_n_bits_single(const uint32_t n, T *out) {
    if (count < n) {
      return false;
    } else {
      get_n_bits_single<T>(n, out);

      // printf("n: %llu out: %lx\n", (unsigned long long) n, (unsigned
      // long)(*out));
      return true;
    }
  }

  __device__ void on_demand_read() {
    count = __shfl_sync(FULL_MASK, count, 0);
    head = __shfl_sync(FULL_MASK, head, 0);

    bool read = (read_bytes_input + threadIdx.x) < len;
    if (read) {
      // update count
      //  uint8_t head_idx = (head + ((count + 7)/8) + threadIdx.x) % buff_len;
      uint8_t head_idx = (head + ((count + 31) / 32) + threadIdx.x) % buff_len;

      *(buff + head_idx) = in[read_bytes_input + threadIdx.x + pointer_off];
      // if(threadIdx.x == 0) printf("read data: %lx\n", in[read_bytes_input +
      // threadIdx.x + pointer_off]);
    }
    uint32_t read_sync = __ballot_sync(FULL_MASK, read);
    uint8_t read_count = __popc(read_sync);
    read_bytes_input += read_count;
    count += (32 * 32);
  }

  __device__ void skip_n_bits(const uint32_t n) {
    fill_buf();
    // if(threadIdx.x == 0) printf("uint_bit_offset: %lu counter: %lu n: %lu
    // head: %lu \n",(unsigned long) uint_bit_offset, (unsigned long) count,
    // (unsigned long)n, (unsigned long)head);

    uint_bit_offset += n;
    if (uint_bit_offset >= 32) {
      uint_bit_offset = uint_bit_offset % 32;
      head = (head + 1) % buff_len;

      count--;
    }
  }
  __device__ void skip_n_bits_single(const uint32_t n) {
    // printf("uint_bit_offset: %lu counter: %lu n: %lu head: %lu \n",(unsigned
    // long) uint_bit_offset, (unsigned long) count, (unsigned long)n, (unsigned
    // long)head);

    uint_bit_offset += n;
    if (uint_bit_offset >= 32) {
      uint_bit_offset = uint_bit_offset % 32;
      head = (head + 1) % buff_len;
      //  printf("update\n");
    }
    count -= n;
  }

  __device__ void align_bits() {
    if (uint_bit_offset % 8 != 0) {
      uint_bit_offset = ((uint_bit_offset + 7) / 8) * 8;
      if (uint_bit_offset == 32) {
        uint_bit_offset = 0;
        head = (head + 1) % buff_len;
        count--;
      }
    }
  }

  template <typename T> __device__ void peek_n_bits(const uint32_t n, T *out) {
    fill_buf();

    get_n_bits<T>(n, out, false);
  }
  template <typename T>
  __device__ bool peek_n_bits_single(const uint32_t n, T *out) {
    // fill_buf();
    if (count < n) {
      return false;
    }
    get_n_bits_single<T>(n, out, false);
    return true;
  }

  __device__ bool check_buf(const uint32_t n) {
    // printf("count%i n: %i\n",(int) count, (int) n);
    if (count < n)
      return false;
    return true;
  }
};
