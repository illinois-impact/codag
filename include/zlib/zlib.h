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
#include <common_warp.h>

#define BUFF_LEN 2


#define UNCOMP 0
#define STATIC 1
#define DYNAMIC 2
#define FULL_MASK 0xFFFFFFFF

#define MAXBITS 15
#define FIXLCODES 288
#define MAXDCODES 30
#define MAXCODES 316

#define LOG2LENLUT 5
#define LOG2DISTLUT 8

#define NMEMCPY 4 // Must be power of 2, acceptable = {1,2,4}
#define NMEMCPYLOG2 2
#define MEMCPYLARGEMASK 0xfffffffc
#define MEMCPYSMALLMASK 0x00000003

static const __device__ __constant__ uint16_t g_lens[29] = {  // Size base for length codes 257..285
  3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
  31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};


static const __device__ __constant__ uint16_t
  g_lext[29] = { 
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};


static const __device__ __constant__ uint16_t
  g_dists[30] = {  // Offset base for distance codes 0..29
    1,   2,   3,   4,   5,   7,    9,    13,   17,   25,   33,   49,   65,    97,    129,
    193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};

static const __device__ __constant__ uint16_t g_dext[30] = {  // Extra bits for distance codes 0..29
  0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};



/// permutation of code length codes
static const __device__ __constant__ uint8_t g_code_order[19 + 1] = {
  16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15, 0xff};



inline __device__ unsigned int bfe(unsigned int source,
                                   unsigned int bit_start,
                                   unsigned int num_bits)
{
  unsigned int bits;
  asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"(source), "r"(bit_start), "r"(num_bits));
  return bits;
};



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


struct s_huffman {
    int16_t lencnt[MAXBITS + 1];
    int16_t off[MAXBITS + 1];
    int16_t distcnt[MAXBITS + 1];
    int16_t distsym[MAXDCODES];
    dynamic_huffman dh;
};

struct inflate_lut{
  int32_t len_lut[1 << LOG2LENLUT];
  int32_t dist_lut[1 << LOG2DISTLUT];
  uint16_t first_slow_len; 
  uint16_t index_slow_len;
  uint16_t first_slow_dist;
  uint16_t index_slow_dist;

};


struct device_space{
    inflate_lut* d_lut;
};


typedef struct __align__(32)
{
    simt::atomic<uint8_t, simt::thread_scope_device>  counter;
    simt::atomic<uint8_t, simt::thread_scope_device>  lock[32];

} __attribute__((aligned (32))) slot_struct;


typedef uint64_t write_queue_ele;

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



template <size_t WRITE_COL_LEN = 512>
struct decompress_output {

    uint8_t* out_ptr;
    uint32_t counter;

    __device__
    decompress_output(uint8_t* ptr, uint64_t CHUNK_SIZE):
        out_ptr(ptr) {
            counter = 0;
            
    }


      template <uint32_t NUM_THREAD = 8>
   // __forceinline__ 
    __device__
    void col_memcpy_div(uint8_t idx, uint16_t len, uint16_t offset, uint8_t div, uint32_t MASK) {
       /*
        IDEAS: N Byte prefix and suffix. Prefix will load value before start. Suffix will write any value at end because it should be done in order anyways. 
       */
        if (len < 32) {
          int tid = threadIdx.x - div * NUM_THREAD;
          uint32_t orig_counter = __shfl_sync(MASK, counter, idx);

          uint8_t num_writes = ((len - tid + NUM_THREAD - 1) / NUM_THREAD);
          uint32_t start_counter =  orig_counter - offset;


          uint32_t read_counter = start_counter + tid;
          uint32_t write_counter = orig_counter + tid;

          if(read_counter >= orig_counter){
              read_counter = (read_counter - orig_counter) % offset + start_counter;
          }

          uint8_t num_ph =  (len +  NUM_THREAD - 1) / NUM_THREAD;
          #pragma unroll 
          for(int i = 0; i < num_ph; i++){
              if(i < num_writes){
                  out_ptr[write_counter + WRITE_COL_LEN * idx] = out_ptr[read_counter + WRITE_COL_LEN * idx];
                  read_counter += NUM_THREAD;
                  write_counter += NUM_THREAD;
              }
              __syncwarp();
          }
      
          //set the counter
          if(threadIdx.x == idx)
              counter += len;
        } else {
            /* Memcpy aligned on write boundaries */
            int tid = threadIdx.x - div * NUM_THREAD;
            uint32_t orig_counter = __shfl_sync(MASK, counter, idx); // This is equal to the tid's counter value upon entering the function

            // prefix aligning bytes needed
            uint8_t prefix_bytes = (uint8_t) (NMEMCPY - (orig_counter & MEMCPYSMALLMASK)) + NMEMCPY;
            // if (prefix_bytes == NMEMCPY) prefix_bytes = 0;
            if (prefix_bytes > len) prefix_bytes = len;

            // suffix aligning bytes needed
            uint8_t suffix_bytes = (uint8_t) ((len - prefix_bytes) & MEMCPYSMALLMASK);
            if (prefix_bytes + suffix_bytes > len) suffix_bytes = len - prefix_bytes;

            // Write prefix and suffix
            uint32_t start_counter =  orig_counter - offset;

            uint32_t read_counter = start_counter + tid;
            uint32_t write_counter = orig_counter + tid;

            // Recruit some threads to instead write the suffix.
            if (threadIdx.x >= prefix_bytes && threadIdx.x < suffix_bytes + prefix_bytes) {
              read_counter += len - suffix_bytes - prefix_bytes;
              write_counter += len - suffix_bytes - prefix_bytes;
            }

            // Check if we need to adjust read location. Happens when offset < len.
            if(read_counter >= orig_counter){
              read_counter = (read_counter - orig_counter) % offset + start_counter;
            }

            // Calculate the number of times this thread has to write an N block.
            uint8_t num_writes = ((len - prefix_bytes - suffix_bytes - tid * NMEMCPY + (NUM_THREAD * NMEMCPY) - 1) / (NUM_THREAD * NMEMCPY));
            if (tid * NMEMCPY + prefix_bytes + suffix_bytes > len) num_writes = 0;

            // Calculate the number of times that the thread with the most blocks has to write.
            uint8_t num_ph =  (len - prefix_bytes - suffix_bytes + (NUM_THREAD * NMEMCPY) - 1) / (NUM_THREAD * NMEMCPY);
            if (prefix_bytes + suffix_bytes > len) num_ph = 0;

            // Write the prefix and the suffix by performing byte memcpy
            if (threadIdx.x < prefix_bytes + suffix_bytes && threadIdx.x < len) {
                out_ptr[write_counter + WRITE_COL_LEN * idx] = out_ptr[read_counter + WRITE_COL_LEN * idx];
            }

            // Change the read and write location to the N byte aligned body.
            read_counter = start_counter + tid * NMEMCPY + prefix_bytes;
            write_counter = orig_counter + tid * NMEMCPY + prefix_bytes;

            // Adjust the read location.
            if(read_counter >= orig_counter){
                read_counter = (read_counter - orig_counter) % offset + start_counter;
            }

            // Recast the output array to a format with N bytes.
            #if NMEMCPY == 2
            uchar2* out_ptr_temp  = reinterpret_cast<uchar2*>(out_ptr);

            #endif
            #if NMEMCPY == 4
            uchar4* out_ptr_temp  = reinterpret_cast<uchar4*>(out_ptr);

            #endif

            __syncwarp();

            //#pragma unroll 
            for(int i = 0; i < num_ph; i++){
                if(i < num_writes){
                    // 1 Byte memcpy
                    #if NMEMCPY == 1
                    out_ptr[write_counter + WRITE_COL_LEN * idx] = out_ptr[read_counter + WRITE_COL_LEN * idx];
                    #endif

                    // 2 Byte memcpy
                    #if NMEMCPY == 2
                    #if DEBUG == 1
                    if ((write_counter + WRITE_COL_LEN * idx) % NMEMCPY != 0)
                      printf("ERROR: Write is not properly aligned at %x\n", write_counter + WRITE_COL_LEN + idx);
                    #endif
                    if (((read_counter + WRITE_COL_LEN * idx) & MEMCPYSMALLMASK) == 0) {
                        out_ptr_temp[(write_counter + WRITE_COL_LEN * idx) >> NMEMCPYLOG2] = out_ptr_temp[(read_counter + WRITE_COL_LEN * idx) >> NMEMCPYLOG2];
                    } else {
                        out_ptr_temp[(write_counter + WRITE_COL_LEN * idx) >> NMEMCPYLOG2] = make_uchar2(out_ptr[read_counter + WRITE_COL_LEN * idx], out_ptr[read_counter + WRITE_COL_LEN * idx + 1]);
                    }
                    // out_ptr[write_counter + WRITE_COL_LEN * idx] = out_ptr[read_counter + WRITE_COL_LEN * idx];
                    // out_ptr[write_counter + WRITE_COL_LEN * idx + 1] = out_ptr[read_counter + WRITE_COL_LEN * idx + 1];
                    #endif

                    // 4 Byte memcpy
                    #if NMEMCPY == 4
                    #if DEBUG == 1
                    if ((write_counter + WRITE_COL_LEN * idx) % NMEMCPY != 0)
                      printf("ERROR: Write is not properly aligned at %x\n", write_counter + WRITE_COL_LEN + idx);
                    #endif
                    if (((read_counter + WRITE_COL_LEN * idx) & MEMCPYSMALLMASK) == 0) {
                        out_ptr_temp[(write_counter + WRITE_COL_LEN * idx) >> NMEMCPYLOG2] = out_ptr_temp[(read_counter + WRITE_COL_LEN * idx) >> NMEMCPYLOG2];
                    } else {
                        out_ptr_temp[(write_counter + WRITE_COL_LEN * idx) >> NMEMCPYLOG2] = make_uchar4(out_ptr[read_counter + WRITE_COL_LEN * idx], out_ptr[read_counter + WRITE_COL_LEN * idx + 1],
                                                                                                         out_ptr[read_counter + WRITE_COL_LEN * idx + 2], out_ptr[read_counter + WRITE_COL_LEN * idx + 3]);
                    }
                    // out_ptr[write_counter + WRITE_COL_LEN * idx] = out_ptr[read_counter + WRITE_COL_LEN * idx];
                    // out_ptr[write_counter + WRITE_COL_LEN * idx + 1] = out_ptr[read_counter + WRITE_COL_LEN * idx + 1];
                    // out_ptr[write_counter + WRITE_COL_LEN * idx + 2] = out_ptr[read_counter + WRITE_COL_LEN * idx + 2];
                    // out_ptr[write_counter + WRITE_COL_LEN * idx + 3] = out_ptr[read_counter + WRITE_COL_LEN * idx + 3];
                    #endif

                    // Adjust the read and write locations.
                    read_counter += NUM_THREAD * NMEMCPY;
                    write_counter += NUM_THREAD * NMEMCPY;
                } 
                __syncwarp();
            }

       
            //__nanosleep(10);

        
            // Increment the counter by len. No race condition because we are using the full warp mask.
            if(threadIdx.x == idx)
                counter += len;
        }
  

    }


    //__forceinline__ 
    __device__
    void write_literal(uint8_t idx, uint8_t b){
        if(threadIdx.x == idx){
            out_ptr[counter + WRITE_COL_LEN * idx] = b;
            counter++;
        }
    }


};


__device__ void init_length_lut(inflate_lut *s_lut, const int16_t* cnt, const int16_t* orig_symbols, int t, uint32_t NUM_THREAD)
{

  int32_t* lut = s_lut -> len_lut;

  for (uint32_t bits = t; bits < (1 << LOG2LENLUT); bits += NUM_THREAD) {
    int sym                = -10 << 5;
    unsigned int first     = 0;
    const int16_t* symbols = orig_symbols;

    unsigned int rbits     = __brev(bits) >> (32 - LOG2LENLUT);
    for (unsigned int len = 1; len <= LOG2LENLUT; len++) {
      unsigned int code  = (rbits >> (LOG2LENLUT - len)) - first;
      unsigned int count = cnt[len];
     
      if (code < count) {
        sym = symbols[code];
   
    
        if (sym > 256) {
        // printf("sym larger\n");
            int lext = g_lext[sym - 257];
          sym = (256 + g_lens[sym - 257]) | (((1 << lext) - 1) << (16 - 5)) | (len << (24 - 5));
          len += lext;
        }
    
        sym = (sym << 5) | len;
        break;
      }
      symbols += count;  // else update for next length
      first += count;
      first <<= 1;
    }
    
    lut[bits] = sym;
    
  }
  if (!t) {
    unsigned int first = 0;
    unsigned int index = 0;
    for (unsigned int len = 1; len <= LOG2LENLUT; len++) {
      unsigned int count = cnt[len];
      index += count;
      first += count;
      first <<= 1;
    }
    s_lut->first_slow_len = first;
    s_lut->index_slow_len = index;
  }

}
__device__ void init_distance_lut(inflate_lut *s_lut, const int16_t* cnt, const int16_t* orig_symbols, int t, uint32_t NUM_THREAD)
{

  int32_t* lut = s_lut -> dist_lut;
  const int16_t* symbols = orig_symbols;

  for (uint32_t bits = t; bits < (1 << LOG2DISTLUT); bits += NUM_THREAD) {
    int sym                = 0;
    unsigned int first     = 0;
    unsigned int rbits     = __brev(bits) >> (32 - LOG2DISTLUT);
    symbols = orig_symbols;
    for (unsigned int len = 1; len <= LOG2DISTLUT; len++) {
      unsigned int code  = (rbits >> (LOG2DISTLUT - len)) - first;
      int count = cnt[len];
     
      if (code < count) {
        sym = symbols[code];
        int dist = symbols[code];
        int dext = g_dext[dist];
        sym      = g_dists[dist] | (dext << 15);
        sym      = (sym << 5) | len;
        break;
      }
      symbols += count;  // else update for next length
      first += count;
      first <<= 1;
    }
    lut[bits] = sym;  
  }

  if (!t) {
    unsigned int first = 0;
    unsigned int index = 0;
    for (unsigned int len = 1; len <= LOG2DISTLUT; len++) {
      unsigned int count = cnt[len];
      index += count;
      first += count;
      first <<= 1;
    }
    s_lut->first_slow_dist = first;
    s_lut->index_slow_dist = index;
  }

}

template <typename READ_COL_TYPE, size_t in_buff_len = 4>
//__forceinline__
 __device__
int32_t decode_len_lut_full_warp (full_warp_input_stream<READ_COL_TYPE, in_buff_len>&  s, const int16_t* const   counts, const int16_t*   syms, inflate_lut* s_lut, uint32_t next32){


    uint32_t next32r = __brev(next32);
    const int16_t*   symbols = syms;

    uint32_t first = s_lut -> first_slow_len;
    #pragma no unroll
    for (int len = LOG2LENLUT + 1; len <= MAXBITS; len++) {
        uint32_t code  = (next32r >> (32 - len)) - first;

        uint16_t count = counts[len];
    if (code < count)
    {
        int32_t sym = symbols[code];
        if(sym > 256){
            sym -= 257;
            int lext = g_lext[sym];
            sym  = 256 + g_lens[sym] + bfe(next32, len, lext);
            len += lext;

        }
        s.skip_n_bits(len);

        return sym;
    }
        symbols += count;
        first += count;
        first <<= 1;
    }
    return -10;
}

template <typename READ_COL_TYPE, size_t in_buff_len = 4>
//__forceinline__
 __device__
uint16_t decode_dist_lut_full_warp (full_warp_input_stream<READ_COL_TYPE, in_buff_len>&  s, const int16_t* const   counts, const int16_t*   syms, inflate_lut* s_lut,  uint32_t next32){




    uint32_t next32r  = __brev(next32);
    const int16_t*   symbols = syms;

    uint32_t first = s_lut -> first_slow_dist;
    #pragma no unroll
    for (int len = LOG2DISTLUT + 1; len <= MAXBITS; len++) {
        uint32_t code  = (next32r >> (32 - len)) - first;

        uint16_t count = counts[len];
    if (code < count)
    {
        int dist = symbols[code];
        int dext = g_dext[dist];

        int off = (g_dists[dist] + bfe(next32, len, dext));
        len += dext;
        s.skip_n_bits(len);
        return (uint16_t)off;
    }
        symbols += count;
        first += count;
        first <<= 1;
    }
    return -10;
}



//Construct huffman tree
template <typename READ_COL_TYPE, size_t in_buff_len = 4>
 __device__ 
void construct_full_warp(full_warp_input_stream<READ_COL_TYPE, in_buff_len>& __restrict__ s, int16_t* const __restrict__ counts , int16_t* const  __restrict__ symbols, 
    const int16_t* const __restrict__ length,  int16_t* const __restrict__ offs, const int num_codes){


    int len;
    //#pragma  unroll
    for(len = 0; len <= MAXBITS; len++){
        counts[len] = 0;
    }

    #pragma no unroll
    for(len = 0; len < num_codes; len++){
        symbols[len] = 0;
        (counts[length[len]])++;
    }
  

    //int16_t offs[16];
    //offs[0] = 0;
    offs[1] = 0;

    #pragma no unroll
    for (len = 1; len < MAXBITS; len++){
        offs[len + 1] = offs[len] + counts[len];
    }

    #pragma no unroll
    for(int16_t symbol = 0; symbol < num_codes; symbol++){
         if (length[symbol] != 0){
            symbols[offs[length[symbol]]++] = symbol;
            //offs[length[symbol]]++;
        }
    }
        
}

template <typename READ_COL_TYPE, size_t in_buff_len = 4>
//__forceinline__
 __device__
int16_t decode_full_warp (full_warp_input_stream<READ_COL_TYPE, in_buff_len>&  s, const int16_t* const   counts, const int16_t*   syms){


    uint32_t next32r = 0;
    s.template peek_n_bits<uint32_t>(32, &next32r);

    next32r = __brev(next32r);
    const int16_t*   symbols = syms;


    uint32_t first = 0;
    #pragma no unroll
    for (uint8_t len = 1; len <= MAXBITS; len++) {
        //if(len == LOG2LENLUT + 1) printf("first: %lu\n", first);

        uint32_t code  = (next32r >> (32 - len)) - first;
        
        uint16_t count = counts[len];
    if (code < count) 
    {
        uint32_t temp = 0;
        s.template fetch_n_bits<uint32_t>(len, &temp);
        //printf("code: %lu count: %lu len: %i, off:%i temp:%lx \n", (unsigned long)code, (unsigned long) count ,(int) len, (int) (symbols - syms), (unsigned long) temp);

    return symbols[code];
    }
        symbols += count;  
        first += count;
        first <<= 1;
    }
    return -10;
}


//construct huffman tree for dynamic huffman encoding block
template <typename READ_COL_TYPE, size_t in_buff_len = 4>
//__forceinline__
 __device__
void decode_dynamic_full_warp(full_warp_input_stream<READ_COL_TYPE, in_buff_len>& s, dynamic_huffman* huff_tree_ptr, const uint32_t buff_idx,
    int16_t* const s_len, int16_t* const s_distcnt, int16_t* const s_distsym, int16_t* const s_off){



    uint16_t hlit;
    uint16_t hdist;
    uint16_t hclen;


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
   // printf("idx: %llu\n", (unsigned long long)buff_idx);
    int16_t* lengths = huff_tree_ptr[buff_idx].treelen;

    for (index = 0; index < 4; index++) {
          lengths[g_code_order[index]] = (int16_t)(temp & 0x07);
            temp >>=3;
    }
    //#pragma no unroll
    for (index = 4; index < hclen; index++) {
        s.template fetch_n_bits<uint32_t>(3, &temp);
        lengths[g_code_order[index]] = (int16_t)temp;
    }

   // #pragma no unroll
    for (; index < 19; index++) {
        lengths[g_code_order[index]] = 0;
    }
       



    construct_full_warp<READ_COL_TYPE, in_buff_len>(s, s_len, huff_tree_ptr[buff_idx].lensym, lengths, s_off, 19);


     index = 0;
     //symbol;
    while (index < hlit + hdist) {
         uint32_t d_temp;
    s.template peek_n_bits<uint32_t>(32, &d_temp);


        int32_t symbol =  decode_full_warp<READ_COL_TYPE, in_buff_len>(s, s_len, huff_tree_ptr[buff_idx].lensym);
  
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

    construct_full_warp<READ_COL_TYPE, in_buff_len>(s, s_len, huff_tree_ptr[buff_idx].lensym, lengths, s_off, hlit);
    construct_full_warp<READ_COL_TYPE, in_buff_len>(s, s_distcnt, s_distsym, (lengths + hlit), s_off, hdist);


    return;
}




template <typename READ_COL_TYPE, size_t in_buff_len = 4, size_t WRITE_COL_LEN>
//__forceinline__ 
__device__ 
void decode_symbol_rdw_lut(full_warp_input_stream<READ_COL_TYPE, in_buff_len>& s, decompress_output<WRITE_COL_LEN>& out, /*const dynamic_huffman* const huff_tree_ptr, unsigned buff_idx,*/
    const int16_t* const s_len, const int16_t* const lensym_ptr, const int16_t* const s_distcnt, const int16_t* const s_distsym, inflate_lut* s_lut) {

    while(1){
        uint32_t next32 = 0;
        uint32_t sym = 0;
        int32_t len_sym = 0;

        s.template peek_n_bits<uint32_t>(32, &next32);
    
        len_sym = (s_lut -> len_lut)[next32 & ((1 << LOG2LENLUT) - 1)];
        uint32_t len = 0;
        if ((uint32_t)len_sym < (uint32_t)(0x100 << 5)) {
            len = len_sym & 0x1f;

            len_sym >>= 5;
            if(threadIdx.x == 0) out.write_literal(0, len_sym);
        
            next32 >>= len;
            s.skip_n_bits(len);
            len_sym = (s_lut -> len_lut)[next32 & ((1 << LOG2LENLUT) - 1)];
        }

        if(len_sym > 0){
            len = len_sym & 0x1f;

            s.skip_n_bits(len);

            sym = ((len_sym >> 5) & 0x3ff) + ((next32 >> (len_sym >> 24)) & ((len_sym >> 16) & 0x1f));
        }

        else{
            sym = decode_len_lut_full_warp<READ_COL_TYPE, in_buff_len>(s,  s_len,  (lensym_ptr) + s_lut -> index_slow_len, s_lut, (next32));
        }

                   

        sym = __shfl_sync(FULL_MASK, sym, 0);

        if(sym <= 255) {
        
             if(threadIdx.x == 0) {out.write_literal(0, sym);  }
        }

        //end of block
        else if(sym == 256) {
            break;
        }

        //lenght, need to parse
        else{
            uint16_t len = 0;
            uint16_t off = 0;
           

     
             uint16_t sym_dist = 0;     
           
            len = (sym & 0x0FFFF) - 256;
            

            next32 = 0;
            s.template peek_n_bits<uint32_t>(32, &next32);
            int dist = (s_lut -> dist_lut)[next32 & ((1 << LOG2DISTLUT) - 1)];
            uint32_t extra_len_dist = 0;


            if(dist > 0){      
              
                uint32_t lut_len = dist & 0x1f;
                int dext = bfe(dist, 20, 5);
                dist = bfe(dist, 5, 15);
                int cur_off = (dist + bfe(next32, lut_len, dext));
                lut_len += dext;

                off = (uint16_t) cur_off;
                s.skip_n_bits(lut_len);

            }
            else{
                off = decode_dist_lut_full_warp<READ_COL_TYPE, in_buff_len>(s,  s_distcnt, s_distsym  + s_lut -> index_slow_dist, s_lut, (next32));    

            }

            out.template col_memcpy_div<32>(0, (uint32_t)len, (uint32_t)off, 0, FULL_MASK);

            //writing       
        }
    }

}

template <typename READ_COL_TYPE, size_t in_buff_len, uint8_t NUM_SUBCHUNKS, size_t WRITE_COL_LEN >
//__forceinline__ 
__device__
void full_warp_shared_lut(full_warp_input_stream<READ_COL_TYPE, in_buff_len>& s, decompress_output<WRITE_COL_LEN>& out_d, uint32_t col_len, uint8_t* out, dynamic_huffman* huff_tree_ptr,
    const fix_huffman* const fixed_tree, int16_t* s_len, int16_t* s_distcnt, int16_t* s_distsym, int16_t* s_off, uint8_t active_chunks, s_huffman* s_tree,
    inflate_lut* s_lut) {
                uint32_t temp;
 s.template peek_n_bits<uint32_t>(32, &temp);

        uint8_t blast = 0;
        uint32_t btype = 0;

        s.template fetch_n_bits<uint32_t>(16, &btype);
       
        btype = 0;

        do{
           s.template fetch_n_bits<uint32_t>(3, &btype);
           blast =  (btype & 0x01);
           btype >>= 1;

            //fixed huffman
            if(btype == 1) {
                __syncwarp();

                init_length_lut (s_lut,   fixed_tree -> lencnt, fixed_tree -> lensym, threadIdx.x, 32);
                init_distance_lut (s_lut,  fixed_tree -> distcnt, fixed_tree -> distsym, threadIdx.x, 32);

                __syncwarp();
                decode_symbol_rdw_lut<READ_COL_TYPE, in_buff_len, WRITE_COL_LEN> (s, out_d,  fixed_tree -> lencnt, fixed_tree -> lensym, fixed_tree -> distcnt, fixed_tree -> distsym, s_lut);
           
            }
            //dyamic huffman
            else if (btype == 0){
                //printf("uncomp block\n");
            }
            else{
                decode_dynamic_full_warp<READ_COL_TYPE, in_buff_len>(s, &(s_tree->dh) ,  0, s_len, s_distcnt, s_distsym, s_off);

                __syncwarp();
                          

                init_length_lut (s_lut,  s_len, (s_tree->dh).lensym, threadIdx.x, 32);
                init_distance_lut (s_lut,  s_distcnt, s_distsym, threadIdx.x, 32);

                __syncwarp();

                decode_symbol_rdw_lut<READ_COL_TYPE, in_buff_len, WRITE_COL_LEN>(s, out_d, s_len, (s_tree->dh).lensym, s_distcnt, s_distsym, s_lut);
                __syncwarp();

            }
        

        }while(blast != 1);

    
    
    __syncwarp(FULL_MASK);

}



template <typename READ_COL_TYPE, typename COMP_COL_TYPE,  uint8_t NUM_SUBCHUNKS, int NUMTHREADS, int NUMBLOCKS, uint16_t in_queue_size = 4, size_t local_queue_size = 4,  size_t WRITE_COL_LEN = 512>
__global__ void 
//__launch_bounds__ (64, 32)
//__launch_bounds__ (96, 21)
//__launch_bounds__ (160, 12)
//__launch_bounds__ (NUMTHREADS, NUMBLOCKS)
__launch_bounds__ (32)

inflate(uint8_t* comp_ptr, const uint64_t* const blk_offset_ptr, uint8_t*out, dynamic_huffman* huff_tree_ptr,
 const fix_huffman* const fixed_tree, uint64_t CHUNK_SIZE, uint64_t num_chunks) {
    
    __shared__ READ_COL_TYPE in_queue_[NUM_SUBCHUNKS][in_queue_size];
    __shared__ s_huffman shared_tree [NUM_SUBCHUNKS];
    __shared__ inflate_lut test_lut [NUM_SUBCHUNKS];
   

    uint8_t active_chunks = NUM_SUBCHUNKS;
    if((blockIdx.x+1) * NUM_SUBCHUNKS > num_chunks){
       active_chunks = num_chunks - blockIdx.x * NUM_SUBCHUNKS;
    }

    __syncthreads();
    int   my_queue = (threadIdx.y);
    int   my_block_idx =  (blockIdx.x * NUM_SUBCHUNKS + threadIdx.y);
    uint64_t    col_len = (blk_offset_ptr[my_block_idx+1] - blk_offset_ptr[my_block_idx]);

    full_warp_input_stream<READ_COL_TYPE, in_queue_size> s(comp_ptr, col_len, blk_offset_ptr[my_block_idx] / sizeof(COMP_COL_TYPE), in_queue_[my_queue], true);

    __syncwarp();
    decompress_output<WRITE_COL_LEN> d((out + CHUNK_SIZE * my_block_idx ), CHUNK_SIZE);

    full_warp_shared_lut<READ_COL_TYPE, in_queue_size, 1, WRITE_COL_LEN>(s, d, (uint32_t) col_len, out, huff_tree_ptr, fixed_tree, shared_tree[my_queue].lencnt ,  shared_tree[my_queue].distcnt, shared_tree[my_queue].distsym , shared_tree[my_queue].off, active_chunks, &(shared_tree[my_queue]), test_lut + my_queue);
}



template <typename READ_COL_TYPE, size_t in_buff_len = 4>
 __device__
int16_t decode_full_warp_fsm (full_warp_input_stream<READ_COL_TYPE, in_buff_len>&  s, const int16_t* const   counts, const int16_t*   syms){


    uint32_t next32r = 0;
    s.template peek_n_bits_single<uint32_t>(32, &next32r);

    next32r = __brev(next32r);
    const int16_t*   symbols = syms;


    uint32_t first = 0;
    #pragma no unroll
    for (uint8_t len = 1; len <= MAXBITS; len++) {
        //if(len == LOG2LENLUT + 1) printf("first: %lu\n", first);

        uint32_t code  = (next32r >> (32 - len)) - first;
        
        uint16_t count = counts[len];
    if (code < count) 
    {
        uint32_t temp = 0;
        s.template fetch_n_bits_single<uint32_t>(len, &temp);

        return symbols[code];
    }
        symbols += count;  
        first += count;
        first <<= 1;
    }
    return -10;
}


//construct huffman tree for dynamic huffman encoding block
template <typename READ_COL_TYPE, size_t in_buff_len = 4>
//__forceinline__
 __device__
void decode_dynamic_full_warp_fsm(full_warp_input_stream<READ_COL_TYPE, in_buff_len>& s, dynamic_huffman* huff_tree_ptr, const uint32_t buff_idx,
    int16_t* const s_len, int16_t* const s_distcnt, int16_t* const s_distsym, int16_t* const s_off){



    uint16_t hlit;
    uint16_t hdist;
    uint16_t hclen;
    uint32_t head_temp;
    int index;
    uint32_t temp;
    int16_t* lengths;
    int32_t symbol;
    int16_t len;

    char state = 'R';
    int Decode_State = 0 ;
    if(threadIdx.x == 0) goto DECODE_0;

    while(1){
        FSM:
            state = __shfl_sync(FULL_MASK, state, 0);

            switch(state){
                case 'R': { 
                    //fill buffer
                    s.on_demand_read();


                    if(threadIdx.x == 0){
                        switch(Decode_State) {
                            case 0: goto DECODE_0;
                            case 1: goto DECODE_1;
                            case 2: goto DECODE_2;
                            case 3: goto DECODE_3;
                            case 4: goto DECODE_4;
                            case 5: goto DECODE_5;
                            case 6: goto DECODE_6;
               

                            default: goto DONE;
                        }
                    }
                    break;
                }

                case 'D': goto DONE;
                default: goto DONE;
            }

    }


    DECODE_0:
      head_temp = 0;
      if(s.template fetch_n_bits_single<uint32_t>(14, &head_temp) == false){
          state = 'R';
          Decode_State = 0;
          goto FSM;
      }

      hlit = (head_temp & (0x1F));
      head_temp >>= 5;
      hdist = (head_temp & (0x1F));
      head_temp >>= 5;
      hclen = (head_temp);

      hlit += 257;
      hdist += 1;
      hclen += 4;
      index = 1;

    //check
     DECODE_1:
       temp= 0;
       if(s.template fetch_n_bits_single<uint32_t>(12, &temp) == false){
            state = 'R';
            Decode_State = 1;
            goto FSM;
        }

     lengths = huff_tree_ptr[buff_idx].treelen;

      for (index = 0; index < 4; index++) {
            lengths[g_code_order[index]] = (int16_t)(temp & 0x07);
              temp >>=3;
      }
    //#pragma no unroll
    for (index = 4; index < hclen; index++) {
      DECODE_2:
        temp=0;
        if(s.template fetch_n_bits_single<uint32_t>(3, &temp) == false){
            state = 'R';
            Decode_State = 2;
            goto FSM;
        }

        lengths[g_code_order[index]] = (int16_t)temp;
    }

   // #pragma no unroll
    for (; index < 19; index++) {
        lengths[g_code_order[index]] = 0;
    }

    
       
    construct_full_warp<READ_COL_TYPE, in_buff_len>(s, s_len, huff_tree_ptr[buff_idx].lensym, lengths, s_off, 19);



     index = 0;
     //symbol;
    while (index < hlit + hdist) {

      //check 32bits are ready
      DECODE_3:
        if(s.check_buf(32) == false){
            state = 'R';
            Decode_State = 3;
            goto FSM;
        }


    
        symbol =  decode_full_warp_fsm<READ_COL_TYPE, in_buff_len>(s, s_len, huff_tree_ptr[buff_idx].lensym);
        
    

        //represent code lengths of 0 - 15
        if(symbol < 16){
            lengths[(index++)] = symbol;
        }

        else{

             len = 0;
            if(symbol == 16) {
                DECODE_4:
                symbol = 0;
                 if( s.template fetch_n_bits_single<int32_t>(2, &symbol) == false){
                    state = 'R';

                    Decode_State = 4;
                    goto FSM;
                }
                 len = lengths[index - 1];  // last length
                 symbol += 3;
            }
            else if(symbol == 17){

                 DECODE_5:
                 symbol = 0;

                 if( s.template fetch_n_bits_single<int32_t>(3, &symbol) == false){

                    state = 'R';
                    Decode_State = 5;
                    goto FSM;
                }
                symbol += 3;
            }
            else if(symbol == 18) {

                 DECODE_6:
                 symbol = 0;
                 if( s.template fetch_n_bits_single<int32_t>(7, &symbol) == false){
                    state = 'R';
                    Decode_State = 6;
                    goto FSM;
                }
                symbol += 11;
            }
       

            while(symbol-- > 0){
                lengths[index++] = (int16_t) len;
            }

        }
    }

    construct_full_warp<READ_COL_TYPE, in_buff_len>(s, s_len, huff_tree_ptr[buff_idx].lensym, lengths, s_off, hlit);
    construct_full_warp<READ_COL_TYPE, in_buff_len>(s, s_distcnt, s_distsym, (lengths + hlit), s_off, hdist);

    state = 'D';
    goto FSM;

    DONE:
      return;
}

template <typename READ_COL_TYPE, size_t in_buff_len = 4>
//__forceinline__
 __device__
int32_t decode_len_lut_full_warp_fsm (full_warp_input_stream<READ_COL_TYPE, in_buff_len>&  s, const int16_t* const   counts, const int16_t*   syms, inflate_lut* s_lut, uint32_t next32, int& skip_len){


    uint32_t next32r = __brev(next32);
    const int16_t*   symbols = syms;

    uint32_t first = s_lut -> first_slow_len;
    #pragma no unroll
    for (int len = LOG2LENLUT + 1; len <= MAXBITS; len++) {
        uint32_t code  = (next32r >> (32 - len)) - first;

        uint16_t count = counts[len];
    if (code < count)
    {
        //s.template fetch_n_bits<uint32_t>(len, &temp);
        int32_t sym = symbols[code];
        if(sym > 256){
            sym -= 257;
            int lext = g_lext[sym];
          //  printf("g_len: %i ", g_lens[sym]);
            sym  = 256 + g_lens[sym] + bfe(next32, len, lext);
           // printf("el: %lu\n", (unsigned long) bfe(next32, len, lext));
            len += lext;

        }
        skip_len = len;
     //   printf("in skip len:%lu\n", (unsigned long) len);
       // s.skip_n_bits(len);

        return sym;
    }
        symbols += count;
        first += count;
        first <<= 1;
    }
    printf("error\n");
    return -10;
}

template <typename READ_COL_TYPE, size_t in_buff_len = 4>
//__forceinline__
 __device__
uint16_t decode_dist_lut_full_warp_fsm (full_warp_input_stream<READ_COL_TYPE, in_buff_len>&  s, const int16_t* const   counts, const int16_t*   syms, inflate_lut* s_lut,  uint32_t next32, int& skip_len){



    uint32_t next32r  = __brev(next32);
    const int16_t*   symbols = syms;

    uint32_t first = s_lut -> first_slow_dist;
    #pragma no unroll
    for (int len = LOG2DISTLUT + 1; len <= MAXBITS; len++) {
        uint32_t code  = (next32r >> (32 - len)) - first;

        uint16_t count = counts[len];
    if (code < count)
    {
        int dist = symbols[code];
        int dext = g_dext[dist];

        int off = (g_dists[dist] + bfe(next32, len, dext));
        len += dext;
        skip_len = len;
        //s.skip_n_bits(len);
        return (uint16_t)off;
    }
        symbols += count;
        first += count;
        first <<= 1;
    }
    return -10;
}





template <typename READ_COL_TYPE, size_t in_buff_len = 4, size_t WRITE_COL_LEN>
//__forceinline__ 
__device__ 
void decode_symbol_rdw_lut_fsm(full_warp_input_stream<READ_COL_TYPE, in_buff_len>& s, decompress_output<WRITE_COL_LEN>& out, /*const dynamic_huffman* const huff_tree_ptr, unsigned buff_idx,*/
    const int16_t* const s_len, const int16_t* const lensym_ptr, const int16_t* const s_distcnt, const int16_t* const s_distsym, inflate_lut* s_lut) {

    uint32_t next32 = 0;
    uint32_t sym = 0;
    int32_t len_sym = 0;
    uint32_t len = 0;
    int skip_len = 0;
    uint16_t off = 0;
    uint16_t sym_dist;
    int dist;
    uint32_t extra_len_dist;
    uint32_t lut_len;
    int dext;
    int cur_off;
    uint16_t mem_len;

    char state = 'R';
    int Decode_State = 0 ;
    if(threadIdx.x == 0) goto START;

    while(1){
        FSM:
            state = __shfl_sync(FULL_MASK, state, 0);

            switch(state){
                case 'R': { 
                    //fill buffer
                    s.on_demand_read();

                    if(threadIdx.x == 0){
                        switch(Decode_State) {
                            case 0: goto DECODE_0;
                            case 1: goto DECODE_1;
                            case 2: goto DECODE_2;
                            case 3: goto DECODE_3;
                            case 4: goto DECODE_4;
                            case 5: goto DECODE_5;
                            case 6: goto DECODE_6;
                            default: goto DONE;
                        }
                    }
                    break;
                }

                case 'W':{
                  mem_len =  __shfl_sync(FULL_MASK, mem_len, 0);
                  off =  __shfl_sync(FULL_MASK, off, 0);
                  out.template col_memcpy_div<32>(0, (uint32_t)mem_len, (uint32_t)off, 0, FULL_MASK);
                                 //   if(threadIdx.x == 0)printf("len: %lu off: %lu\n", (unsigned long) mem_len, (unsigned long) off);


                  if(threadIdx.x == 0)
                    goto START;

                  break;
                }

                case 'D': goto DONE;
                default: goto DONE;
            }
    }

    while(1){

      START:
        next32 = 0;
        sym = 0;
        len_sym = 0;

      DECODE_0:
        if (s.template peek_n_bits_single<uint32_t>(32, &next32)==false){

           state = 'R';
           Decode_State = 0;
           goto FSM;
        }

        
       // printf("cur next32: %lx\n", next32);
        

        len_sym = (s_lut -> len_lut)[next32 & ((1 << LOG2LENLUT) - 1)];
        len = 0;

        if ((uint32_t)len_sym < (uint32_t)(0x100 << 5)) {
            len = len_sym & 0x1f;

            len_sym >>= 5;
            if(threadIdx.x == 0) out.write_literal(0, len_sym);

            next32 >>= len;
            DECODE_1:
              if(s.check_buf(len) ==false){

                 state = 'R';
                 Decode_State = 1;
                 goto FSM;
              }
            s.skip_n_bits_single((uint32_t)len);
                                  //  printf("len: %i", len);

            len_sym = (s_lut -> len_lut)[next32 & ((1 << LOG2LENLUT) - 1)];
        }


        if(len_sym > 0){
            len = len_sym & 0x1f;
            DECODE_2:
              if(s.check_buf(len) == false){
                 state = 'R';
                 Decode_State = 2;
                 goto FSM;
              }
            s.skip_n_bits_single(len);
                     //   printf("len2: %i", len);

            sym = ((len_sym >> 5) & 0x3ff) + ((next32 >> (len_sym >> 24)) & ((len_sym >> 16) & 0x1f));
        }

        else{
            skip_len = 0;
            sym = decode_len_lut_full_warp_fsm<READ_COL_TYPE, in_buff_len>(s,  s_len,  (lensym_ptr) + s_lut -> index_slow_len, s_lut, (next32), skip_len);
           // printf("skip len: %i\n", skip_len);
            DECODE_3:
              if(s.check_buf(skip_len) == false){
                 printf("jump\n");
                 state = 'R';
                 Decode_State = 3;
                 goto FSM;
              }
             s.skip_n_bits_single(skip_len);

        }
       // printf("check2");

                   
        if(sym <= 255) {
        
             if(threadIdx.x == 0) { out.write_literal(0, sym);  }
        }

        //end of block
        else if(sym == 256) {
            state = 'D';
            goto FSM;
        }

        //lenght, need to parse
        else{
             mem_len = 0;
             off = 0;
             sym_dist = 0;     
             mem_len = (sym & 0x0FFFF) - 256;
             next32 = 0;
      
            DECODE_4:
              if(s.check_buf(32) == false){
                 state = 'R';
                 Decode_State = 4;
                 goto FSM;
              }
            s.template peek_n_bits_single<uint32_t>(32, &next32);

            dist = (s_lut -> dist_lut)[next32 & ((1 << LOG2DISTLUT) - 1)];
            extra_len_dist = 0;


            if(dist > 0){     
               // printf("next32:%lx\n", next32);               
                lut_len = dist & 0x1f;
                dext = bfe(dist, 20, 5);
                dist = bfe(dist, 5, 15);
                cur_off = (dist + bfe(next32, lut_len, dext));
                lut_len += dext;
                off = (uint16_t) cur_off;

                DECODE_5:
                  if(s.check_buf(lut_len) == false){
                     state = 'R';
                     Decode_State = 5;
                     goto FSM;
                  }
                s.skip_n_bits_single(lut_len);
                //printf("off 1\n");
            }
            else{
                skip_len = 0;
                off = decode_dist_lut_full_warp_fsm<READ_COL_TYPE, in_buff_len>(s,  s_distcnt, s_distsym  + s_lut -> index_slow_dist, s_lut, (next32), skip_len);    

                DECODE_6:
                  if(s.check_buf(skip_len) == false){
                     state = 'R';
                     Decode_State = 6;
                     goto FSM;
                  }
                 s.skip_n_bits_single(skip_len);
                                // printf("off 2\n");

            }

               

            state = 'W';
            goto FSM;

                  
            //writing       
        }
       // printf("check3");

    }

    DONE:
      return;
}



template <typename READ_COL_TYPE, size_t in_buff_len, uint8_t NUM_SUBCHUNKS, size_t WRITE_COL_LEN >
__device__
void full_warp_shared_lut_fsm(full_warp_input_stream<READ_COL_TYPE, in_buff_len>& s, decompress_output<WRITE_COL_LEN>& out_d, uint32_t col_len, uint8_t* out, dynamic_huffman* huff_tree_ptr,
     const fix_huffman* const fixed_tree, int16_t* s_len, int16_t* s_distcnt, int16_t* s_distsym, int16_t* s_off, uint8_t active_chunks, s_huffman* s_tree,
    inflate_lut* s_lut) {

            uint32_t temp;
            bool r;


            if(threadIdx.x == 0){
               if(s.check_buf(32) == false){
                  r= true;     
                }
                else r=false;
            }
            else {r = false;}
            r = __shfl_sync(FULL_MASK, r, 0);
            if(r)  {s.on_demand_read();}

           if(threadIdx.x == 0) {s.template peek_n_bits_single<uint32_t>(32, &temp); }

     
           uint8_t blast = 0;
           uint32_t btype = 0;

          if(threadIdx.x == 0){
               if(s.check_buf(16) == false){
                  r= true;     
                }
                else
                  r=false;
            }
            else {r = false;}
          
          r = __shfl_sync(FULL_MASK, r, 0);
          if(r)  {s.on_demand_read();}

         if(threadIdx.x == 0) s.template fetch_n_bits_single<uint32_t>(16, &btype);

        btype = 0;

        do{
            r = false;
            if(threadIdx.x == 0){
             if(s.check_buf(3) == false){
                    r= true;
              }
                else r=false;

            }
            r = __shfl_sync(FULL_MASK, r, 0);
            if(r)  {s.on_demand_read();}

            btype = 0;
            if(threadIdx.x == 0) s.template fetch_n_bits_single<uint32_t>(3, &btype);
             btype = __shfl_sync(FULL_MASK, btype, 0);

           blast =  (btype & 0x01);
           btype >>= 1;

            //fixed huffman
            if(btype == 1) {
                //decode_dynamic_full_warp_fsm<READ_COL_TYPE, in_buff_len>(s, &(s_tree->dh) ,  0, s_len, s_distcnt, s_distsym, s_off);
                __syncwarp();

                init_length_lut (s_lut,   fixed_tree -> lencnt, fixed_tree -> lensym, threadIdx.x, 32);
                init_distance_lut (s_lut,  fixed_tree -> distcnt, fixed_tree -> distsym, threadIdx.x, 32);

                __syncwarp();
                decode_symbol_rdw_lut_fsm<READ_COL_TYPE, in_buff_len, WRITE_COL_LEN> (s, out_d,  fixed_tree -> lencnt, fixed_tree -> lensym, fixed_tree -> distcnt, fixed_tree -> distsym, s_lut);
           

            }
            //dyamic huffman
            else if (btype == 0){
                printf("uncomp block\n");
            }
            else{
       


                decode_dynamic_full_warp_fsm<READ_COL_TYPE, in_buff_len>(s, &(s_tree->dh) ,  0, s_len, s_distcnt, s_distsym, s_off);

                __syncwarp();
              
                init_length_lut (s_lut,  s_len, (s_tree->dh).lensym, threadIdx.x, 32);
                init_distance_lut (s_lut,  s_distcnt, s_distsym, threadIdx.x, 32);

                __syncwarp();

                decode_symbol_rdw_lut_fsm<READ_COL_TYPE, in_buff_len, WRITE_COL_LEN>(s, out_d, s_len, (s_tree->dh).lensym, s_distcnt, s_distsym, s_lut);
                __syncwarp();


            }
        

        }while(blast != 1);

    
    
    __syncwarp(FULL_MASK);

}


template <typename READ_COL_TYPE, typename COMP_COL_TYPE,  uint8_t NUM_SUBCHUNKS, int NUMTHREADS, int NUMBLOCKS, uint16_t in_queue_size = 4, size_t local_queue_size = 4,  size_t WRITE_COL_LEN = 512>
__global__ void 
//__launch_bounds__ (64, 32)
//__launch_bounds__ (96, 21)
//__launch_bounds__ (160, 12)
//__launch_bounds__ (NUMTHREADS, NUMBLOCKS)
__launch_bounds__ (32)

inflate_shared_rdw_fsm(uint8_t* comp_ptr, const uint64_t* const blk_offset_ptr, uint8_t*out, dynamic_huffman* huff_tree_ptr,
  const fix_huffman* const fixed_tree, uint64_t CHUNK_SIZE, uint64_t num_chunks) {
    
    __shared__ READ_COL_TYPE in_queue_[NUM_SUBCHUNKS][in_queue_size];

    __shared__ s_huffman shared_tree [NUM_SUBCHUNKS];

    __shared__ inflate_lut test_lut [NUM_SUBCHUNKS];
   

    uint8_t active_chunks = NUM_SUBCHUNKS;
    if((blockIdx.x+1) * NUM_SUBCHUNKS > num_chunks){
       active_chunks = num_chunks - blockIdx.x * NUM_SUBCHUNKS;
    }

    __syncthreads();
    int   my_queue = (threadIdx.y);
    int   my_block_idx =  (blockIdx.x * NUM_SUBCHUNKS + threadIdx.y) ;
    uint64_t    col_len = (blk_offset_ptr[my_block_idx+1] - blk_offset_ptr[my_block_idx]);

    full_warp_input_stream<READ_COL_TYPE, in_queue_size> s(comp_ptr, col_len, blk_offset_ptr[my_block_idx] / sizeof(COMP_COL_TYPE), in_queue_[my_queue]);

    __syncwarp();
    decompress_output<WRITE_COL_LEN> d((out + CHUNK_SIZE * my_block_idx ), CHUNK_SIZE);

    full_warp_shared_lut_fsm<READ_COL_TYPE, in_queue_size, 1, WRITE_COL_LEN>(s, d, (uint32_t) col_len, out, huff_tree_ptr, fixed_tree, shared_tree[my_queue].lencnt ,  shared_tree[my_queue].distcnt, shared_tree[my_queue].distsym , shared_tree[my_queue].off, active_chunks, &(shared_tree[my_queue]), test_lut + my_queue);
}


namespace deflate {

template <typename READ_COL_TYPE, size_t WRITE_COL_LEN, uint16_t queue_depth, uint8_t NUM_SUBCHUNKS, bool All_Thread_Decoding = true>
 __host__ void decompress_gpu(const uint8_t* const in, uint8_t** out, const uint64_t in_n_bytes, uint64_t* out_n_bytes,
  uint64_t* blk_offset_f, const uint64_t blk_n_bytes, uint64_t chunk_size) {

    uint64_t num_blk = ((uint64_t) blk_n_bytes / sizeof(uint64_t)) - 2;
    uint64_t data_size = blk_offset_f[0];

    uint8_t* d_in;
    uint64_t* d_blk_offset;
    uint8_t* d_out;

    //fix
    int num_sm = 108;
   
    cuda_err_chk(cudaMalloc(&d_in, in_n_bytes));
    cuda_err_chk(cudaMalloc(&d_blk_offset, blk_n_bytes));

    cuda_err_chk(cudaMemcpy(d_in, in, in_n_bytes, cudaMemcpyHostToDevice));
    cuda_err_chk(cudaMemcpy(d_blk_offset, blk_offset_f+1, blk_n_bytes, cudaMemcpyHostToDevice));



    uint64_t out_bytes = chunk_size * num_blk;

    *out_n_bytes = data_size;
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

    memcpy(f_tree.lencnt, fix_lencnt, sizeof(uint16_t)*16);
    memcpy(f_tree.lensym, fix_lensym, sizeof(uint16_t)*FIXLCODES);
    memcpy(f_tree.distcnt, fix_distcnt, sizeof(uint16_t)*(MAXBITS + 1));
    memcpy(f_tree.distsym, fix_distsym, sizeof(uint16_t)*MAXDCODES);

    cuda_err_chk(cudaMemcpy(d_f_tree, &f_tree, sizeof(fix_huffman), cudaMemcpyHostToDevice));


    dim3 blockD(32,3,1);
    // dim3 blockD2(32,2,1);
    // dim3 blockD1(32,1,1);

    //num_blk = 1;
   // uint64_t num_tblk = (num_blk + NUM_SUBCHUNKS - 1) / NUM_SUBCHUNKS;
    uint64_t num_tblk = (num_blk ) / NUM_SUBCHUNKS;


    dim3 gridD(num_tblk,1,1);
    dim3 blockD2(32,1,1);


    cudaDeviceSynchronize();



    std::chrono::high_resolution_clock::time_point kernel_start = std::chrono::high_resolution_clock::now();

    if(All_Thread_Decoding){
        inflate<uint32_t, READ_COL_TYPE, 1, 256, 8, 64 , 4, WRITE_COL_LEN> <<<num_blk,32>>> (d_in,  d_blk_offset, d_out, d_tree, d_f_tree, chunk_size, num_blk);
    }

    else{
        inflate_shared_rdw_fsm<uint32_t, READ_COL_TYPE, 1, 256, 8, 64 , 4, WRITE_COL_LEN> <<<num_blk,blockD2>>> (d_in,  d_blk_offset, d_out, d_tree, d_f_tree, chunk_size, num_blk);
    }

    cuda_err_chk(cudaDeviceSynchronize());
    std::chrono::high_resolution_clock::time_point kernel_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total = std::chrono::duration_cast<std::chrono::duration<double>>(kernel_end - kernel_start);
    std::cout << "\t" << total.count() << std::endl;


    cudaError_t error = cudaGetLastError();
      if(error != cudaSuccess)
      {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
      }


    *out = new uint8_t[data_size];
    cuda_err_chk(cudaMemcpy((*out), d_out, data_size, cudaMemcpyDeviceToHost));
    cuda_err_chk(cudaFree(d_out));
    cuda_err_chk(cudaFree(d_in));
    cuda_err_chk(cudaFree(d_blk_offset));
 }


}

//#endif // __ZLIB_H__


