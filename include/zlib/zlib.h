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


struct s_huffman {
    int16_t lencnt[MAXBITS + 1];
    int16_t off[MAXBITS + 1];
    int16_t distcnt[MAXBITS + 1];
    int16_t distsym[MAXDCODES];
    dynamic_huffman dh;
};



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
        //#pragma unroll 
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






template <typename READ_COL_TYPE, typename COMP_COL_TYPE, uint8_t NUM_SUBCHUNKS>
//__forceinline__ 
__device__
void reader_warp(decompress_input<READ_COL_TYPE, COMP_COL_TYPE>& in, queue<READ_COL_TYPE>& rq, uint8_t active_chunks) {
    //iterate number of chunks for the single reader warp
   int t = 0;
   while(true){
        bool done = true;
        for(uint8_t cur_chunk = 0; cur_chunk < active_chunks; cur_chunk++){
            COMP_COL_TYPE v;
            uint8_t rc = comp_read_data_seq(FULL_MASK, &v, in, cur_chunk);
            if(rc != 0)
                done = false;

       

            warp_enqueue<COMP_COL_TYPE, READ_COL_TYPE>(&v, &rq, cur_chunk, rc);
        }
        if(done)
            break;
    }

}






template <typename READ_COL_TYPE, size_t in_buff_len = 4>
//__forceinline__
 __device__
int16_t decode (input_stream<READ_COL_TYPE, in_buff_len>&  s, const int16_t* const   counts, const int16_t*   symbols){

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




/// permutation of code length codes
static const __device__ __constant__ uint8_t g_code_order[19 + 1] = {
  16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15, 0xff};


//construct huffman tree for dynamic huffman encoding block
template <typename READ_COL_TYPE, size_t in_buff_len = 4>
//__forceinline__
 __device__
void decode_dynamic(input_stream<READ_COL_TYPE, in_buff_len>& s, dynamic_huffman* huff_tree_ptr, const uint32_t buff_idx,
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

    construct<READ_COL_TYPE, in_buff_len>(s, s_len, huff_tree_ptr[buff_idx].lensym, lengths, s_off, hlit);
    construct<READ_COL_TYPE, in_buff_len>(s, s_distcnt, s_distsym, (lengths + hlit), s_off, hdist);


    return;
}




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




template <typename READ_COL_TYPE, size_t in_buff_len = 4>
//__forceinline__ 
__device__ 
void decode_symbol(input_stream<READ_COL_TYPE, in_buff_len>& s, queue<write_queue_ele>& mq, /*const dynamic_huffman* const huff_tree_ptr, unsigned buff_idx,*/
    const int16_t* const s_len, const int16_t* const lensym_ptr, const int16_t* const s_distcnt, const int16_t* const s_distsym) {

    uint64_t c = 0;

    while(1){
;

        uint16_t sym = decode<READ_COL_TYPE, in_buff_len>(s,  s_len,  lensym_ptr);

        if(sym <= 255) {
            write_queue_ele qe;
            qe.type = 0;
            qe.data = (uint32_t) sym;
            mq.enqueue(&qe);
            
            c++;
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
 
            uint32_t extra_len_dist = 0;
            if(g_dext[sym_dist] != 0){
                s.template fetch_n_bits<uint32_t>(g_dext[sym_dist], &extra_len_dist);
            }


            write_queue_ele qe;
            qe.data = (len << 16) | (extra_len_dist + g_dists[sym_dist]);
            qe.type = 1;
            c+=len;
            mq.enqueue(&qe);            
        }
    }

}



template <typename READ_COL_TYPE, size_t in_buff_len, uint8_t NUM_SUBCHUNKS>
//__forceinline__ 
__device__
void decoder_warp(input_stream<READ_COL_TYPE, in_buff_len>& s,  queue<write_queue_ele>& mq, uint32_t col_len, uint8_t* out, dynamic_huffman* huff_tree_ptr,
    slot_struct* d_slot_struct, const fix_huffman* const fixed_tree, int16_t* s_len, int16_t* s_distcnt, int16_t* s_distsym, int16_t* s_off, uint8_t active_chunks) {

    unsigned sm_id;

    uint8_t slot = 0;
    if(threadIdx.x == 0){
       sm_id = get_smid();
       slot = find_slot(sm_id, d_slot_struct);
    }

    slot = __shfl_sync(FULL_MASK, slot, 0);
    sm_id = __shfl_sync(FULL_MASK, sm_id, 0);

    if(threadIdx.x >= active_chunks){}

    else{

        uint8_t blast;
        uint32_t btype;

      
        do{

        s.template fetch_n_bits<uint32_t>(19, &btype);

        
        btype >>= 16;
        blast =  (btype & 0x01);
        btype >>= 1;
        //fixed huffman
        if(btype == 1) {
            decode_symbol<READ_COL_TYPE, in_buff_len> (s, mq,  fixed_tree -> lencnt, fixed_tree -> lensym, fixed_tree -> distcnt, fixed_tree -> distsym);
        }
        //dyamic huffman
        else if (btype == 0){
            printf("uncomp\n");
             s.template align_bits();
             int32_t uncomp_len;
             s.template fetch_n_bits<int32_t>(16, &uncomp_len);
             s.template fetch_n_bits<uint32_t>(16, &btype);

            for(int32_t c = 0; c < uncomp_len; c++){
                s.template fetch_n_bits<uint32_t>(8, &btype);
            }
             
             break;
        }
        else{

            decode_dynamic<READ_COL_TYPE, in_buff_len>(s, huff_tree_ptr,  (uint32_t)((sm_id * 32 + slot) * 32 + threadIdx.x), s_len, s_distcnt, s_distsym, s_off);
            decode_symbol<READ_COL_TYPE, in_buff_len>(s, mq, 
                s_len, huff_tree_ptr[((sm_id * 32 + slot) * 32 + threadIdx.x)].lensym, s_distcnt, s_distsym);

        }

        }while(blast != 1);

    }

    __syncwarp(FULL_MASK);
    if(threadIdx.x == 0){
        release_slot(sm_id, slot, d_slot_struct);
    }

}



template <typename READ_COL_TYPE, size_t in_buff_len, uint8_t NUM_SUBCHUNKS>
//__forceinline__ 
__device__
void decoder_warp_shared(input_stream<READ_COL_TYPE, in_buff_len>& s,  queue<write_queue_ele>& mq, uint32_t col_len, uint8_t* out, dynamic_huffman* huff_tree_ptr,
    slot_struct* d_slot_struct, const fix_huffman* const fixed_tree, int16_t* s_len, int16_t* s_distcnt, int16_t* s_distsym, int16_t* s_off, uint8_t active_chunks, s_huffman* s_tree) {

   

    if(threadIdx.x >= active_chunks){}

    else{

        uint8_t blast;
        uint32_t btype;
        s.template fetch_n_bits<uint32_t>(16, &btype);
        btype = 0;

        do{

        s.template fetch_n_bits<uint32_t>(3, &btype);
      //  btype >>= 16;
        blast =  (btype & 0x01);
        btype >>= 1;
        //fixed huffman
        if(btype == 1) {
            decode_symbol<READ_COL_TYPE, in_buff_len> (s, mq,  fixed_tree -> lencnt, fixed_tree -> lensym, fixed_tree -> distcnt, fixed_tree -> distsym);
        }
        //dyamic huffman
        else if (btype == 0){
		printf("uncomp\n");
         // s.template align_bits();
         // int32_t uncomp_len;
         // s.template fetch_n_bits<int32_t>(16, &uncomp_len);
         // s.template fetch_n_bits<uint32_t>(16, &btype);

         // for(int32_t c = 0; c < uncomp_len; c++){
         //    s.template fetch_n_bits<uint32_t>(8, &btype);
         // }
         // break;
        }
        else{

            decode_dynamic<READ_COL_TYPE, in_buff_len>(s, &(s_tree->dh) ,  0, s_len, s_distcnt, s_distsym, s_off);
            decode_symbol<READ_COL_TYPE, in_buff_len>(s, mq, 
                s_len, (s_tree->dh).lensym, s_distcnt, s_distsym);

        }

        }while(blast != 1);

    }

    __syncwarp(FULL_MASK);


}




template <size_t WRITE_COL_LEN, uint8_t NUM_SUBCHUNKS>
__device__
void writer_warp_8div(queue<write_queue_ele>& mq, decompress_output<WRITE_COL_LEN>& out, uint64_t CHUNK_SIZE ) {
    int div = 0;
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


    uint32_t done = 0;
    while (!done) {

        bool deq = false;
        //uint64_t v = 0;
        write_queue_ele v;
        if(threadIdx.x < NUM_SUBCHUNKS)
            mq.attempt_dequeue(&v, &deq);
        uint32_t deq_mask = __ballot_sync(MASK, deq);
        uint32_t deq_count = __popc(deq_mask);


        for (size_t i = 0; i < deq_count; i++) {
            int32_t f = __ffs(deq_mask);
            uint8_t t = __shfl_sync(MASK, v.type, f-1);
            uint32_t d = __shfl_sync(MASK, v.data, f-1);

            //pair
            if(t == 1){
                uint64_t len = d >> 16;
                uint64_t offset = (d) & 0x0000ffff;
                out.template col_memcpy_div<4>(f-1, (uint32_t)len, (uint32_t)offset, div, MASK);
            }
            //literal
            else{
                uint8_t b = (d) & 0x00FF;
                out.write_literal(f-1, b);
            }

            deq_mask >>= f;
            deq_mask <<= f;
        }
        bool check = out.counter != (CHUNK_SIZE );
        if(threadIdx.x != 0 ) check = false;
        done = __ballot_sync(MASK, check) == 0;
    } 

}
template <size_t WRITE_COL_LEN, uint8_t NUM_SUBCHUNKS>
__device__
void writer_warp(queue<write_queue_ele>& mq, decompress_output<WRITE_COL_LEN>& out, uint64_t CHUNK_SIZE, uint8_t active_chunks ) {

    uint32_t done = 0;
    while (!done) {

        bool deq = false;
        write_queue_ele v;
        if(threadIdx.x < active_chunks){
            mq.attempt_dequeue(&v, &deq);
        }
        uint32_t deq_mask = __ballot_sync(FULL_MASK, deq);
        uint32_t deq_count = __popc(deq_mask);


        for (size_t i = 0; i < deq_count; i++) {
            int32_t f = __ffs(deq_mask);
            uint8_t t = __shfl_sync(FULL_MASK, v.type, f-1);
            uint32_t d = __shfl_sync(FULL_MASK, v.data, f-1);

            //pair
            if(t == 1){
                uint64_t len = d >> 16;
                uint64_t offset = (d) & 0x0000ffff;
                out.template col_memcpy_div<32>(f-1, (uint32_t)len, (uint32_t)offset, 0, FULL_MASK);
            }
            //literal
            else{
                uint8_t b = (d) & 0x00FF;
                out.write_literal(f-1, b);
            }

            deq_mask >>= f;
            deq_mask <<= f;
       	    
       	}
        __syncwarp();
        bool check = out.counter != (CHUNK_SIZE);
        if(threadIdx.x >= active_chunks ) check = false;
        done = __ballot_sync(FULL_MASK, check) == 0;
    } 

}

template <size_t WRITE_COL_LEN = 512>
//__forceinline__ 
__device__
void writer_warp_8div_warp2(queue<write_queue_ele>& mq, decompress_output<WRITE_COL_LEN>& out, int division, uint64_t CHUNK_SIZE) {
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
        done = __ballot_sync(MASK, (out.counter != (CHUNK_SIZE / 32)) && (threadIdx.x % 2 == division)) == 0;


    } 

}



template <typename READ_COL_TYPE, typename COMP_COL_TYPE,  uint8_t NUM_SUBCHUNKS, uint16_t in_queue_size = 4, size_t out_queue_size = 4, size_t local_queue_size = 4,  size_t WRITE_COL_LEN = 512>
__global__ void 
//__launch_bounds__ (96, 13)
//__launch_bounds__ (96, 32)
inflate(uint8_t* comp_ptr, const uint64_t* const col_len_ptr, const uint64_t* const blk_offset_ptr, uint8_t*out, dynamic_huffman* huff_tree_ptr,
 slot_struct* d_slot_struct, const fix_huffman* const fixed_tree, uint64_t CHUNK_SIZE, uint64_t num_chunks) {
    __shared__ READ_COL_TYPE in_queue_[NUM_SUBCHUNKS][in_queue_size];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> h[NUM_SUBCHUNKS];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> t[NUM_SUBCHUNKS];

    __shared__ write_queue_ele out_queue_[NUM_SUBCHUNKS][out_queue_size];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> out_h[NUM_SUBCHUNKS];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> out_t[NUM_SUBCHUNKS];

    __shared__ READ_COL_TYPE local_queue[NUM_SUBCHUNKS][local_queue_size];

    //MAXDCODES is 30
    __shared__ int16_t s_distsym[NUM_SUBCHUNKS][MAXDCODES];
    __shared__ int16_t s_off[NUM_SUBCHUNKS][16];
    __shared__ int16_t s_lencnt[NUM_SUBCHUNKS][16];
    __shared__ int16_t s_distcnt[NUM_SUBCHUNKS][16];

    //initialize heads and tails to be 0
    if(threadIdx.x < NUM_SUBCHUNKS){
        h[threadIdx.x] = 0;
        t[threadIdx.x] = 0;
        out_h[threadIdx.x] = 0;
        out_t[threadIdx.x] = 0;
    }

    __syncthreads();

    uint8_t active_chunks = NUM_SUBCHUNKS;
    if((blockIdx.x+1) * NUM_SUBCHUNKS > num_chunks){
       active_chunks = num_chunks - blockIdx.x * NUM_SUBCHUNKS;
       //printf("bid: %i actie chunks: %u\n", blockIdx.x, active_chunks);
    }

    int my_block_idx = blockIdx.x * NUM_SUBCHUNKS + threadIdx.x % NUM_SUBCHUNKS;
    int my_queue = threadIdx.x % NUM_SUBCHUNKS;
    uint64_t col_len = (col_len_ptr[my_block_idx]);

    if (threadIdx.y == 0) {
        queue<READ_COL_TYPE> in_queue(in_queue_[my_queue], h + my_queue , t + my_queue, in_queue_size);
        decompress_input<READ_COL_TYPE, COMP_COL_TYPE> d(comp_ptr, col_len, blk_offset_ptr[my_block_idx] / sizeof(COMP_COL_TYPE));
        reader_warp<READ_COL_TYPE, COMP_COL_TYPE, NUM_SUBCHUNKS>(d, in_queue, active_chunks);
   	printf("tid: %i reading done\n", threadIdx.x);
    }

    else if (threadIdx.y == 1) {
        queue<READ_COL_TYPE> in_queue(in_queue_[my_queue], h + my_queue, t + my_queue, in_queue_size);
        queue<write_queue_ele> out_queue(out_queue_[my_queue], out_h + my_queue, out_t + my_queue, out_queue_size);
        input_stream<READ_COL_TYPE, local_queue_size> s(&in_queue, (uint32_t)col_len, local_queue[my_queue], threadIdx.x < active_chunks);
        decoder_warp<READ_COL_TYPE, local_queue_size, NUM_SUBCHUNKS>(s, out_queue, (uint32_t) col_len, out, huff_tree_ptr, d_slot_struct, fixed_tree, s_lencnt[my_queue], s_distcnt[my_queue], s_distsym[my_queue], s_off[my_queue], active_chunks);
	printf("tid: %i decoding done\n", threadIdx.x);
    }

    else {
        queue<write_queue_ele> out_queue(out_queue_[my_queue], out_h + my_queue, out_t + my_queue, out_queue_size);
        decompress_output<WRITE_COL_LEN> d((out + CHUNK_SIZE * blockIdx.x * NUM_SUBCHUNKS), CHUNK_SIZE);
        writer_warp<WRITE_COL_LEN, NUM_SUBCHUNKS>(out_queue, d, CHUNK_SIZE, active_chunks);

    }

    __syncthreads();


}

template <typename READ_COL_TYPE, typename COMP_COL_TYPE,  uint8_t NUM_SUBCHUNKS, uint16_t in_queue_size = 4, size_t out_queue_size = 4, size_t local_queue_size = 4,  size_t WRITE_COL_LEN = 512>
__global__ void 
//__launch_bounds__ (96, 13)
//__launch_bounds__ (96, 32)
inflate_shared(uint8_t* comp_ptr, const uint64_t* const col_len_ptr, const uint64_t* const blk_offset_ptr, uint8_t*out, dynamic_huffman* huff_tree_ptr,
 slot_struct* d_slot_struct, const fix_huffman* const fixed_tree, uint64_t CHUNK_SIZE, uint64_t num_chunks) {
    
    __shared__ READ_COL_TYPE in_queue_[NUM_SUBCHUNKS][in_queue_size];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> h[NUM_SUBCHUNKS];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> t[NUM_SUBCHUNKS];

    __shared__ write_queue_ele out_queue_[NUM_SUBCHUNKS][out_queue_size];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> out_h[NUM_SUBCHUNKS];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> out_t[NUM_SUBCHUNKS];

    __shared__ READ_COL_TYPE local_queue[NUM_SUBCHUNKS][local_queue_size];

    __shared__ s_huffman shared_tree [NUM_SUBCHUNKS];


    //initialize heads and tails to be 0
    if(threadIdx.x < NUM_SUBCHUNKS){
        h[threadIdx.x] = 0;
        t[threadIdx.x] = 0;
        out_h[threadIdx.x] = 0;
        out_t[threadIdx.x] = 0;
    }

    __syncthreads();

    uint8_t active_chunks = NUM_SUBCHUNKS;
    if((blockIdx.x+1) * NUM_SUBCHUNKS > num_chunks){
       active_chunks = num_chunks - blockIdx.x * NUM_SUBCHUNKS;
    }

    int my_block_idx = blockIdx.x * NUM_SUBCHUNKS + threadIdx.x % NUM_SUBCHUNKS;
    int my_queue = threadIdx.x % NUM_SUBCHUNKS;
    //uint64_t col_len = (col_len_ptr[my_block_idx]);
    uint64_t col_len = (blk_offset_ptr[my_block_idx+1] - blk_offset_ptr[my_block_idx]);

    if (threadIdx.y == 0) {
        queue<READ_COL_TYPE> in_queue(in_queue_[my_queue], h + my_queue , t + my_queue, in_queue_size);
        decompress_input<READ_COL_TYPE, COMP_COL_TYPE> d(comp_ptr, col_len, blk_offset_ptr[my_block_idx] / sizeof(COMP_COL_TYPE));
        reader_warp<READ_COL_TYPE, COMP_COL_TYPE, NUM_SUBCHUNKS>(d, in_queue, active_chunks);
    }

    else if (threadIdx.y == 1) {
        queue<READ_COL_TYPE> in_queue(in_queue_[my_queue], h + my_queue, t + my_queue, in_queue_size);
        queue<write_queue_ele> out_queue(out_queue_[my_queue], out_h + my_queue, out_t + my_queue, out_queue_size);
        input_stream<READ_COL_TYPE, local_queue_size> s(&in_queue, (uint32_t)col_len, local_queue[my_queue], threadIdx.x < active_chunks);
        decoder_warp_shared<READ_COL_TYPE, local_queue_size, NUM_SUBCHUNKS>(s, out_queue, (uint32_t) col_len, out, huff_tree_ptr, d_slot_struct, fixed_tree, shared_tree[my_queue].lencnt ,  shared_tree[my_queue].distcnt, shared_tree[my_queue].distsym , shared_tree[my_queue].off, active_chunks, &(shared_tree[my_queue]));

    }

    else {
        queue<write_queue_ele> out_queue(out_queue_[my_queue], out_h + my_queue, out_t + my_queue, out_queue_size);
        decompress_output<WRITE_COL_LEN> d((out + CHUNK_SIZE * blockIdx.x * NUM_SUBCHUNKS), CHUNK_SIZE);
        writer_warp<WRITE_COL_LEN, NUM_SUBCHUNKS>(out_queue, d, CHUNK_SIZE, active_chunks);
    }

    __syncthreads();


}

namespace deflate {

template <typename READ_COL_TYPE, size_t WRITE_COL_LEN, uint16_t queue_depth, uint8_t NUM_SUBCHUNKS, bool shared_tree_flag = false>
 __host__ void decompress_gpu(const uint8_t* const in, uint8_t** out, const uint64_t in_n_bytes, uint64_t* out_n_bytes,
  uint64_t* col_len_f, const uint64_t col_n_bytes, uint64_t* blk_offset_f, const uint64_t blk_n_bytes, uint64_t chunk_size) {

    uint64_t num_blk = ((uint64_t) blk_n_bytes / sizeof(uint64_t)) - 2;

    uint64_t data_size = blk_offset_f[0];


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
    cuda_err_chk(cudaMemcpy(d_blk_offset, blk_offset_f+1, blk_n_bytes, cudaMemcpyHostToDevice));



    uint64_t out_bytes = chunk_size * num_blk;
    std::cout << (int)NUM_SUBCHUNKS << "\t" << chunk_size << "\t" << WRITE_COL_LEN << "\t" << queue_depth << "\t"  << in_n_bytes << "\t" << blk_n_bytes + col_n_bytes;
    uint8_t* d_out;
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

    //f_tree.lencnt = fix_lencnt;


    memcpy(f_tree.lencnt, fix_lencnt, sizeof(uint16_t)*16);
    memcpy(f_tree.lensym, fix_lensym, sizeof(uint16_t)*FIXLCODES);
    memcpy(f_tree.distcnt, fix_distcnt, sizeof(uint16_t)*(MAXBITS + 1));
    memcpy(f_tree.distsym, fix_distsym, sizeof(uint16_t)*MAXDCODES);

    cuda_err_chk(cudaMemcpy(d_f_tree, &f_tree, sizeof(fix_huffman), cudaMemcpyHostToDevice));


    dim3 blockD(32,3,1);
    num_blk = 1;
    uint64_t num_tblk = (num_blk + NUM_SUBCHUNKS - 1) / NUM_SUBCHUNKS;

    dim3 gridD(num_tblk,1,1);
    cudaDeviceSynchronize();

    std::chrono::high_resolution_clock::time_point kernel_start = std::chrono::high_resolution_clock::now();

    if(shared_tree_flag){
        inflate_shared<uint64_t, READ_COL_TYPE, NUM_SUBCHUNKS, queue_depth , queue_depth, 4, WRITE_COL_LEN> <<<gridD,blockD>>> (d_in, d_col_len, d_blk_offset, d_out, d_tree, d_slot_struct, d_f_tree, chunk_size, num_blk);
    }
    else{
        inflate<uint64_t, READ_COL_TYPE, NUM_SUBCHUNKS, queue_depth , queue_depth, 4, WRITE_COL_LEN> <<<gridD,blockD>>> (d_in, d_col_len, d_blk_offset, d_out, d_tree, d_slot_struct, d_f_tree, chunk_size, num_blk);
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
    cuda_err_chk(cudaFree(d_col_len));
    cuda_err_chk(cudaFree(d_blk_offset));
 }


}

//#endif // __ZLIB_H__
