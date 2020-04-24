
#include <common.h>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <fstream>

#include <unistd.h>
#include <iostream>
#include <cstring>
#include <unistd.h>
#include <sys/types.h> 
#include <sys/stat.h> 
#include <fcntl.h>
#include <sys/mman.h>
#include <chrono>


constexpr   uint16_t THRDS_SM_() { return (2048); }
constexpr   uint16_t BLK_SIZE_() { return (32); }
constexpr   uint16_t BLKS_SM_()  { return (THRDS_SM_()/BLK_SIZE_()); }
constexpr   uint64_t GRID_SIZE_() { return (1024); }
constexpr   uint64_t NUM_CHUNKS_() { return (GRID_SIZE_()*BLK_SIZE_()); }
constexpr   uint64_t CHUNK_SIZE_() { return (4*1024*32); }
constexpr   uint64_t HEADER_SIZE_() { return (1); }
constexpr   uint32_t OVERHEAD_PER_CHUNK_(uint32_t d) { return (ceil<uint32_t>(d,(HEADER_SIZE_()*8))+1); } 
constexpr   uint32_t HIST_SIZE_() { return 2048; }
constexpr   uint32_t LOOKAHEAD_SIZE_() { return 512; }
constexpr   uint32_t REF_SIZE_() { return 16; }
constexpr   uint32_t REF_SIZE_BYTES_() { return REF_SIZE_()/8; }
constexpr   uint32_t OFFSET_SIZE_() { return (bitsNeeded((uint32_t)HIST_SIZE_())); }
constexpr   uint32_t LENGTH_SIZE_() { return (REF_SIZE_()-OFFSET_SIZE_()); }
constexpr   uint32_t LENGTH_MASK_(uint32_t d) { return ((d > 0) ? 1 | (LENGTH_MASK_(d-1)) << 1 : 0);  }
constexpr   uint32_t MIN_MATCH_LENGTH_() { return (ceil<uint32_t>((OFFSET_SIZE_()+LENGTH_SIZE_()),8)+1); }
constexpr   uint32_t MAX_MATCH_LENGTH_() { return (pow<uint32_t, uint32_t>(2,LENGTH_SIZE_()) + MIN_MATCH_LENGTH_() - 1); }
constexpr   uint8_t DEFAULT_CHAR_() { return ' '; }
constexpr   uint32_t HEAD_INTS_() { return 7; }
constexpr   uint32_t READ_UNITS_() { return 4; }
constexpr   uint32_t LOOKAHEAD_UNITS_() { return LOOKAHEAD_SIZE_()/READ_UNITS_(); }
constexpr   uint64_t WARP_ID_(uint64_t t) { return t/32; }
constexpr   uint32_t LOOKAHEAD_SIZE_4_BYTES_() { return  LOOKAHEAD_SIZE_()/sizeof(uint32_t); }
constexpr   uint32_t HIST_SIZE_4_BYTES_() { return  HIST_SIZE_()/sizeof(uint32_t); }
constexpr   uint32_t INPUT_BUFFER_SIZE() { return (8); }


#define BLKS_SM                           BLKS_SM_()
#define THRDS_SM                          THRDS_SM_()
#define BLK_SIZE			  BLK_SIZE_()			  
#define GRID_SIZE			  GRID_SIZE_()			  
#define NUM_CHUNKS			  NUM_CHUNKS_()
#define CHUNK_SIZE                        CHUNK_SIZE_()
#define HEADER_SIZE			  HEADER_SIZE_()			  
#define OVERHEAD_PER_CHUNK(d)       	  OVERHEAD_PER_CHUNK_(d)	  
#define HIST_SIZE			  HIST_SIZE_()			  
#define LOOKAHEAD_SIZE			  LOOKAHEAD_SIZE_()			  
#define OFFSET_SIZE			  OFFSET_SIZE_()			  
#define LENGTH_SIZE			  LENGTH_SIZE_()
#define LENGTH_MASK(d)			  LENGTH_MASK_(d)   
#define MIN_MATCH_LENGTH		  MIN_MATCH_LENGTH_()		  
#define MAX_MATCH_LENGTH		  MAX_MATCH_LENGTH_()		  
#define DEFAULT_CHAR			  DEFAULT_CHAR_()			  
#define HEAD_INTS                         HEAD_INTS_()
#define READ_UNITS                        READ_UNITS_()
#define LOOKAHEAD_UNITS                   LOOKAHEAD_UNITS_()
#define WARP_ID(t)                        WARP_ID_(t)
#define INPUT_BUFFER_SIZE                 INPUT_BUFFER_SIZE()

namespace lzss {
    __host__ __device__ void find_match(const uint8_t* const  hist, const uint32_t hist_head, const uint32_t hist_count, const uint8_t* const lookahead, const uint32_t lookahead_head, const uint32_t lookahead_count, uint32_t* offset, uint32_t* length) {
	uint32_t hist_offset = 1;
	uint32_t f_length = 0;
	uint32_t max_len = 0;
	uint32_t max_offset = 0;
	while (hist_offset < hist_count) {
	    if (hist[((hist_head+hist_offset) % HIST_SIZE)] == lookahead[(lookahead_head)]) {
		f_length = 1;

		while (((hist_offset + f_length) < hist_count) && ((f_length) < lookahead_count) &&
		       (hist[((hist_head + hist_offset + f_length) % HIST_SIZE)] ==
			lookahead[((lookahead_head + f_length) % LOOKAHEAD_SIZE)])) {
		    f_length++;
		    if (f_length >= MAX_MATCH_LENGTH)
			break;
		}
		if (f_length > max_len) {
		    max_len = f_length;
		    max_offset = hist_offset;
		}
	    }

	    if (f_length >= MAX_MATCH_LENGTH) {
		f_length = MAX_MATCH_LENGTH;
		max_len = f_length;
		break;
	    }

	    hist_offset++;
	    

	    
	}
	if (max_len < MIN_MATCH_LENGTH)
	    max_len = 0;
	*length = max_len;
	*offset = hist_count - max_offset;

    }


	__device__ void decompress_func_new(const uint8_t* const in, uint8_t* out, const uint64_t in_n_bytes, const uint64_t out_n_bytes, const uint64_t out_chunk_size, const uint64_t n_chunks, const uint64_t* const blk_off, const uint64_t* const col_len, const uint8_t* const col_map) {
		uint8_t hist[HIST_SIZE] = {DEFAULT_CHAR};
		uint64_t hist_head = 0;
		uint64_t hist_count = 0;
		int tid = threadIdx.x;
		int chunk_idx = blockIdx.x;
		uint64_t used_bytes = 0;
		uint64_t mychunk_size = col_len[blockDim.x * chunk_idx + tid];
		uint64_t in_start_idx = blk_off[chunk_idx];

		uint8_t header_byte = 0;
		uint8_t block = 0;

		uint64_t out_bytes = 0;
		uint64_t out_start_idx = chunk_idx * CHUNK_SIZE;

		uint8_t compress_counter = 0;
		uint64_t col_idx = col_map[blockDim.x * chunk_idx + tid];

		uint8_t v = 0;
		uint32_t* in_4B = (uint32_t *)(&(in[in_start_idx]));
		uint32_t in_4B_off = 0;


		int data_counter = 0;

		uint32_t out_buffer = 0;
		uint8_t* out_buffer_8 = (uint8_t*) &out_buffer;
		uint8_t out_buffer_tail = 0;
		uint64_t out_off = 0;

		uint32_t* out_4B = (uint32_t*)(&(out[out_start_idx + col_idx*4]));

		uint8_t input_buffer[INPUT_BUFFER_SIZE];
		uint8_t input_buffer_head = 0;
		uint8_t input_buffer_tail = 0;
		uint8_t input_buffer_count = 0;
		uint8_t input_buffer_read_count = 0;

		bool stall_flag = false;
		bool header_active = false;
		uint8_t header_off = 0;
		bool type = 0;
		uint32_t type_0_byte = 0;
		uint64_t type_0_v = 0;
		while (used_bytes < mychunk_size) {
			unsigned mask = __activemask();
			int res = __popc(mask);

			//read data
			uint32_t* input_buffer_4B = (uint32_t *)(&(input_buffer[input_buffer_tail]));
			input_buffer_4B[0] = in_4B[in_4B_off + tid];
			input_buffer_tail = (input_buffer_tail + 4) % INPUT_BUFFER_SIZE;
			input_buffer_count += 4;


			in_4B_off += res;

			if (!header_active) {
				header_byte = input_buffer[input_buffer_head];
				input_buffer_head = (input_buffer_head + 1) % INPUT_BUFFER_SIZE;
				input_buffer_count--;
				used_bytes++;
				header_off = 0;
				header_active = true;
			}

			for (; header_off < 8; header_off++) {
				type = header_byte & 0x01;

				if (type == 0) {
					uint32_t n_b = MIN_MATCH_LENGTH - 1;

					for (; type_0_byte < n_b; type_0_byte++) {
						if (input_buffer_count) {
							uint64_t k = ((uint64_t)input_buffer[(input_buffer_head)]) << (type_0_byte*8);
							input_buffer_head = (input_buffer_head + 1) % INPUT_BUFFER_SIZE;
							input_buffer_count--;
							type_0_v |= k;
							used_bytes++;
						}
						else
							break;
					}

					if (type_0_byte < n_b)
						break;
					uint32_t length = (type_0_v & LENGTH_MASK(LENGTH_SIZE)) + MIN_MATCH_LENGTH;
					uint32_t offset = type_0_v >> LENGTH_SIZE;
					uint32_t hist_tail = hist_head + hist_count;
					uint32_t start_o = hist_tail - offset;
					for (size_t j = 0; j < length; j++) {
						uint8_t val = hist[(start_o + j) % HIST_SIZE];
						hist[(hist_tail + j) % HIST_SIZE] = val;
						out_buffer_8[out_buffer_tail++] = val;
						if (out_buffer_tail == 4) {
							out_4B[out_off] = out_buffer;
							out_buffer = 0;
							out_off+=32;

							out_buffer_tail = 0;
						}
					}
					hist_count += length;
					if (hist_count > HIST_SIZE) {
						hist_head = (hist_head + (hist_count-HIST_SIZE)) % HIST_SIZE;
						hist_count = HIST_SIZE;
					}

					type_0_v = 0;
					type_0_byte = 0;
					header_byte >>= 1;

				}
				else {
					if (input_buffer_count) {
						uint8_t val = input_buffer[(input_buffer_head)];
						input_buffer_head = (input_buffer_head + 1) % INPUT_BUFFER_SIZE;
						input_buffer_count--;
						used_bytes++;
						out_buffer_8[out_buffer_tail++] = val;
						if (out_buffer_tail == 4) {
							out_4B[out_off] = out_buffer;
							out_buffer = 0;
							out_off+=32;

							out_buffer_tail = 0;
						}
						hist[(hist_head+hist_count)%HIST_SIZE] = val;
						hist_count += 1;
						if (hist_count > HIST_SIZE) {
							hist_head = (hist_head + (1)) % HIST_SIZE;
							hist_count = HIST_SIZE;
						}

						header_byte >>= 1;
					}
					else
						break;

				}
				if (used_bytes >= mychunk_size)
					break;
			}

			if (header_off == 8)
				header_active = false;



			__syncwarp(mask);
		}
		
		if (out_buffer_tail) {
			out_4B[out_off] = out_buffer;
		}


    }
    __host__ __device__ void decompress_func(const uint8_t* const in, uint8_t* out, const uint64_t in_n_bytes, const uint64_t out_n_bytes, const uint64_t out_chunk_size, const uint64_t n_chunks, const uint64_t* const lens, const uint64_t tid) {
	if (tid < n_chunks) {
	    //uint8_t out_[CHUNK_SIZE];
	    
	    uint64_t in_start_idx = (tid == 0) ? 0 : lens[tid-1];
	    uint64_t in_end_idx = 	lens[tid];
	    uint64_t my_chunk_size = in_end_idx - in_start_idx;
	
	    uint64_t out_start_idx = tid * CHUNK_SIZE;

	    //uint8_t hist[HIST_SIZE] = {DEFAULT_CHAR};
	    uint8_t lookahead[LOOKAHEAD_SIZE];
	    uint32_t hist_head  = 0;
	    uint32_t hist_count = 0;
	    uint32_t lookahead_head = 0;
	    uint32_t lookahead_count = 0;
	    uint64_t consumed_bytes = 0;
	    uint64_t out_bytes = 0;
	    //uint32_t cur_header_byte_pos = 0;
	    //uint8_t header_byte = 0;
	    //uint8_t blocks = 0;
	    //uint32_t c = 0;
	    uint32_t rem = out_n_bytes % out_chunk_size;
	    uint64_t expected_out = ((tid == (n_chunks-1)) && (rem)) ? rem : out_chunk_size;
	    uint64_t used_bytes = 0;
	    while (used_bytes < my_chunk_size) {
		while ((lookahead_count < LOOKAHEAD_SIZE) && (consumed_bytes < my_chunk_size))  {
		    lookahead[(lookahead_head + (lookahead_count++)) % LOOKAHEAD_SIZE] =
			in[in_start_idx + (consumed_bytes++)];
		}
		uint8_t header = lookahead[(lookahead_head) % LOOKAHEAD_SIZE];
		lookahead_head = (lookahead_head + 1) % LOOKAHEAD_SIZE;
		//if ((tid == 0) && (used_bytes < 10))
		//printf("1: %p\n", header);
		lookahead_count--;
		used_bytes++;
		for (size_t i = 0; (i < 8) && (used_bytes < my_chunk_size); i++, header>>=1) {
		    uint8_t type = header & 0x01;
		    if (type == 0) {
			uint32_t n_b = MIN_MATCH_LENGTH - 1;

			uint64_t v = 0;
			for (size_t j = 0; j < n_b; j++) {
			    uint64_t k = ((uint64_t)lookahead[(lookahead_head)]) << (j*8);
			    //printf("k: %llu\n", (unsigned long long) (k >>(j*8)));
			    v |= k;
			    lookahead_head = (lookahead_head + 1) % LOOKAHEAD_SIZE;
			    lookahead_count--;
			}
			uint32_t length = (v & LENGTH_MASK(LENGTH_SIZE)) + MIN_MATCH_LENGTH;
			uint32_t offset = v >> LENGTH_SIZE;
			uint64_t out_bytes_ = out_bytes; 
			uint64_t out_copy_start = out_bytes_ - offset + out_start_idx;
			for (size_t j = 0; (j < length) ; j++) {
			    out[out_start_idx + out_bytes++] = out[out_copy_start+j];
			    //if ((tid == 0) && (used_bytes < 100))
			    //printf("b: %llu\t1: %c\t ub: %llu\tleng: %llu\toffset: %llu\tj: %llu\tv: %p\n", (unsigned long long) out_bytes, (char)z, (unsigned long long) used_bytes, (unsigned long long) length, (unsigned long long) offset, (unsigned long long) j,v);
			}
			used_bytes += n_b;
			
		    }
		    else {
			uint8_t v = lookahead[(lookahead_head)];
			//if ((tid == 0) && (used_bytes < 100))
			//printf("b: %llu\t1: %c\t ub: %llu\n", (unsigned long long) out_bytes, (char)v, (unsigned long long) used_bytes);
			out[out_start_idx + out_bytes++] = v;
			lookahead_head = (lookahead_head + 1) % LOOKAHEAD_SIZE;
			lookahead_count--;
			used_bytes++;
		    }
		}
	    }

	}
    }




bool vector_comp(const std::pair<uint64_t, uint8_t> &a,  
               const std::pair<uint64_t, uint8_t> &b) { 
       return (a.first > b.first); 
} 


void cpu_compress_func(const uint8_t* const in, uint8_t* out, const uint64_t in_n_bytes, const uint64_t out_n_bytes, 
					   const uint64_t in_chunk_size, const uint64_t out_chunk_size, const uint64_t n_chunks, uint64_t* len_out,  uint8_t* col_map, uint64_t* blk_offset, uint64_t* col_offset, uint64_t* chunk_offset_array) {

	blk_offset[0] = 0;
	
	chunk_offset_array[0] = 0;

	bool test_flag[BLK_SIZE] = {false};
   

	uint8_t hist[BLK_SIZE][HIST_SIZE] = {DEFAULT_CHAR};
    uint8_t lookahead[BLK_SIZE][LOOKAHEAD_SIZE] = {DEFAULT_CHAR};

    uint32_t hist_head[BLK_SIZE]  = {0};
    uint32_t hist_count[BLK_SIZE] = {0};
    uint32_t lookahead_head[BLK_SIZE] = {0};
    uint32_t lookahead_count[BLK_SIZE] = {0};
    uint64_t consumed_bytes[BLK_SIZE] = {0};
    uint64_t out_bytes = 1;
    uint64_t cur_header_byte_pos = 0;
    uint8_t header_byte = 0;
    uint8_t blocks = 0;
    uint64_t used_bytes = 0;
    
    uint64_t in_off[BLK_SIZE] = {0};
	uint8_t in_flag[BLK_SIZE] = {0};

	//int64_t my_chunk_size = (in_n_bytes / BLK_SIZE);
	int64_t my_chunk_size = (CHUNK_SIZE / BLK_SIZE);


    uint64_t in_start_idx;
	uint64_t out_start_idx;

	uint64_t chunk_offset = 0;
	uint64_t chunk_offset2 = 0;

	//lens[BLK_SIZE] = {0};
	//<len, col>


	printf("Num chunks: %i\n", NUM_CHUNKS);
	printf("size chunks: %i\n",CHUNK_SIZE );
	printf("in_chunk_size: %llu\n", in_chunk_size);

	int chunk_num = in_n_bytes / CHUNK_SIZE;

  	printf("chunk num: %i\n", chunk_num);
	for(int chunk_id = 0; chunk_id < chunk_num; chunk_id++){

		if(chunk_id % 100 == 0){
			printf("chunk_id: %i\n", chunk_id);
		}
		//check
		col_offset[chunk_id*BLK_SIZE] = 0;

//checkcol_offset[0] = 0;
		in_start_idx = chunk_id * in_chunk_size;
		out_start_idx = chunk_offset2;


		

		uint64_t padding_counter[BLK_SIZE + 1] = {0};
		std::pair<uint64_t, uint8_t> lens[BLK_SIZE];

	    out_bytes = 1;
	    cur_header_byte_pos = 0;
	    header_byte = 0;
	    blocks = 0;
	    used_bytes = 0;
   

	    for(int k = 0; k  < BLK_SIZE; k++){
	    	hist_head[k]  = 0;
		    hist_count[k] = 0;
		    lookahead_head[k] = 0;
		    lookahead_count[k] = 0;
		    consumed_bytes[k] = 0;
		    in_off[k] = 0;
		    in_flag[k] = 0;



	    }
 		
	 //    out_bytes = 1;
	 //    cur_header_byte_pos = 0;
	 //    header_byte = 0;
	 //    blocks = 0;
	 //    used_bytes = 0;

	 //    in_off[BLK_SIZE] = {0};
	 //    in_flag[BLK_SIZE] = {0};

		// padding_counter[BLK_SIZE + 1] = {0};



    	for(uint8_t i = 0; i < BLK_SIZE; i++){
    		lens[i].first = 0;
    		lens[i].second = i;
    	}

	    for(int t_idx = 0; t_idx < BLK_SIZE; t_idx++){


	    	cur_header_byte_pos = out_bytes - 1;


	    	blocks = 0;
			used_bytes = 0;
	    	
	    	while(used_bytes < my_chunk_size){
	    		//printf("used_bytes: %llu\n", used_bytes);

				while ((lookahead_count[t_idx] < LOOKAHEAD_SIZE) && (consumed_bytes[t_idx] < my_chunk_size))  {
			
						uint8_t temp_v = in[in_start_idx + (consumed_bytes[t_idx]++) + in_off[t_idx] + t_idx*4];

						lookahead[t_idx][(lookahead_head[t_idx] + (lookahead_count[t_idx]++)) % LOOKAHEAD_SIZE] = temp_v;
						//in[in_start_idx + (consumed_bytes[t_idx]++) + itr_off * 4 * 31 + t_idx * 4];
					
					

						in_flag[t_idx]++;
						if(in_flag[t_idx] == 4){
							in_off[t_idx] += 31 * 4;
							in_flag[t_idx] = 0;
						}
				}

				//printf("update lookhead\n");

			


				uint32_t offset = 0;
				uint32_t length = 0;
				find_match(hist[t_idx], hist_head[t_idx], hist_count[t_idx], lookahead[t_idx], lookahead_head[t_idx], lookahead_count[t_idx], &offset, &length);

				if (length == 0) {
				    uint8_t v = lookahead[t_idx][lookahead_head[t_idx]];

		
				    out[out_start_idx + (out_bytes++)] = v;
				    lookahead_head[t_idx] = (lookahead_head[t_idx] + 1) % LOOKAHEAD_SIZE;
				    lookahead_count[t_idx]--;
				    hist[t_idx][(hist_head[t_idx] + hist_count[t_idx])%HIST_SIZE] = v;
				    hist_count[t_idx] += 1;
				    if (hist_count[t_idx] > HIST_SIZE) {
					hist_head[t_idx] = (hist_head[t_idx] + (1)) % HIST_SIZE;
					hist_count[t_idx] = HIST_SIZE;
				    }
				    header_byte = (header_byte | (1 << blocks));
				    used_bytes++;
				    lens[t_idx].first++;

				}

				else {

					// if(out_bytes == 0){
					// 	printf("out_bytes: %llu, v: %c\n", out_bytes, lookahead[t_idx][lookahead_head[t_idx]]);

					// }

				    uint64_t v = (offset << LENGTH_SIZE) | (length - MIN_MATCH_LENGTH);
				    uint64_t v2 = v;
				    //Writing the pair?
				    for (size_t i = 0; i < (MIN_MATCH_LENGTH-1); i++) {
					uint8_t k = v & 0x00FF;
					out[out_start_idx + (out_bytes++)] = k;
					v >>= 8;

					//to compute the length
					lens[t_idx].first++;
				    }
				    //update hist
				    uint32_t hist_start = hist_head[t_idx] + hist_count[t_idx];
				    for (size_t i = 0; i < length; i++) {
					uint8_t z = lookahead[t_idx][(lookahead_head[t_idx]+i) % LOOKAHEAD_SIZE];
					hist[t_idx][(hist_start+i) % HIST_SIZE]= z;
					}
				    
				    lookahead_head[t_idx] = (lookahead_head[t_idx] + length) % LOOKAHEAD_SIZE;
				    lookahead_count[t_idx] -= length;
				    hist_count[t_idx] += length;
				    if (hist_count[t_idx] > HIST_SIZE) {
					hist_head[t_idx] = (hist_head[t_idx] + (hist_count[t_idx]-HIST_SIZE)) % HIST_SIZE;
					hist_count[t_idx] = HIST_SIZE;
				    }
				    used_bytes += length;
				}

				if ((++blocks) == 8) {

					if(chunk_id == 0 && test_flag[t_idx] == false){
						printf("t_idx: %i, header_byte: %x\n", t_idx, header_byte );
						test_flag[t_idx] = true;
					

					} 

					
				    out[out_start_idx + cur_header_byte_pos] = header_byte;
				    header_byte = 0;
				    cur_header_byte_pos = out_bytes++;
				    lens[t_idx].first++;
				    blocks = 0;
				}

				 if (blocks != 0) {
					out[out_start_idx + cur_header_byte_pos] = header_byte;

		     		}

	    	}

	    }

	
	
	    for(int i = 0; i < BLK_SIZE - 1; i++){
	    	//lens[i].first += padding_counter[i+1];
	    	if(i == 0){
	    		col_offset[BLK_SIZE * chunk_id] = 0;
	    	}
	    	else{
	    		col_offset[BLK_SIZE * chunk_id + i + 1] = col_offset[BLK_SIZE * chunk_id + i] + lens[i].first;

	    	}
			//col_offset[BLK_SIZE * chunk_id + i + 1] += padding_counter[i+1];
	    	
	    }
	    



		std::sort(lens, lens + BLK_SIZE, vector_comp);

		uint64_t chunk_len = 0;
		for(int i = 0; i < BLK_SIZE; i++){
       		chunk_len += lens[i].first;
			len_out[chunk_id*BLK_SIZE+i] = lens[i].first;
			col_map[chunk_id*BLK_SIZE+i] = lens[i].second;


		}


		blk_offset[chunk_id+1] = chunk_len;

		chunk_offset2 += ((chunk_len + 127)/128) * 128;
		//check the constant
		//chunk_offset += ceil(chunk_len / 128) * 128;
		chunk_offset += chunk_len;
		chunk_offset_array[chunk_id+1] = chunk_offset;
		//chunk_offset += chunk_len;

	
	}

	   
}




__global__ void parallel_scan(uint64_t* blk_offset, uint64_t n){

	for(int i = 1; i <= n;i++){
		blk_offset[i] += blk_offset[i-1];
	}



}



__global__ void ExampleKernel(uint64_t* col_len, uint8_t* col_map)
{
    // Specialize BlockRadixSort for a 1D block of 128 threads owning 4 integer keys and values each
    //typedef cub::BlockRadixSort<int, 128, 4, int> BlockRadixSort;
    typedef cub::BlockRadixSort<uint64_t, 32, 1, uint8_t> BlockRadixSort;
    // Allocate shared memory for BlockRadixSort
    __shared__ typename BlockRadixSort::TempStorage temp_storage;
    // Obtain a segment of consecutive items that are blocked across threads
    //int thread_keys[4];
    //int thread_values[4];
    
    uint64_t thread_keys[1];
    uint8_t thread_values[1];
    int chunk_idx = blockIdx.x;
    int col_idx = threadIdx.x;

    thread_keys[0] = col_len[BLK_SIZE * chunk_idx + col_idx];
    thread_values[0] = col_idx;


    // Collectively sort the keys and values among block threads
    BlockRadixSort(temp_storage).Sort(thread_keys, thread_values);

    col_len[BLK_SIZE * chunk_idx + blockDim.x - col_idx - 1] = thread_keys[0];
    col_map[BLK_SIZE * chunk_idx + blockDim.x - col_idx - 1] = thread_values[0];

}









__global__  void gpu_compress_init(const uint8_t* const in, const uint64_t in_n_bytes,  
					   					  const uint64_t in_chunk_size, const uint64_t out_chunk_size, const uint64_t n_chunks, 
					                      uint64_t* col_len, uint64_t* blk_offset){


	__shared__ unsigned long long int block_len;

	if(threadIdx.x == 0){
		block_len = 0;
		if(blockIdx.x == 0){
			blk_offset[0] = 0;
		}
	}
	__syncthreads();


	uint8_t hist[HIST_SIZE] = {DEFAULT_CHAR};
    uint8_t lookahead[LOOKAHEAD_SIZE] = {DEFAULT_CHAR};

    uint32_t hist_head  = {0};
    uint32_t hist_count = {0};
    uint32_t lookahead_head = {0};
    uint32_t lookahead_count = {0};
    uint64_t consumed_bytes = {0};
    uint64_t out_bytes = 1;
    uint64_t cur_header_byte_pos = 0;
    uint8_t header_byte = 0;
    uint8_t blocks = 0;
    uint64_t used_bytes = 0;
    
    uint64_t in_off = {0};
	uint8_t in_flag = {0};

	int64_t my_chunk_size = (CHUNK_SIZE / BLK_SIZE);

	uint64_t byte_counter = 1;
	uint64_t byte_offset = 1;
	uint64_t col_counter = BLK_SIZE - 1;


    uint64_t in_start_idx;
    //change
	uint64_t chunk_offset = 0;

	int t_idx = threadIdx.x;
	int chunk_idx = blockIdx.x;

	int col_idx  = threadIdx.x;
	uint64_t out_start_idx = blk_offset[chunk_idx];


	in_start_idx = blockIdx.x * in_chunk_size;


	uint64_t cur_len = 0;


	while(used_bytes < my_chunk_size){

		while ((lookahead_count < LOOKAHEAD_SIZE) && (consumed_bytes < my_chunk_size))  {
	
				uint8_t temp_v = in[in_start_idx + (consumed_bytes++) + in_off + col_idx*4];

				lookahead[(lookahead_head + (lookahead_count++)) % LOOKAHEAD_SIZE] = temp_v;			

				in_flag++;
				if(in_flag== 4){
					in_off += 31 * 4;
					in_flag = 0;
				}


			
		}




		uint32_t offset = 0;
		uint32_t length = 0;
		find_match(hist, hist_head, hist_count, lookahead, lookahead_head, lookahead_count, &offset, &length);

		if (length == 0) {

		    uint8_t v = lookahead[lookahead_head];

   			cur_len++;
 
		    lookahead_head = (lookahead_head + 1) % LOOKAHEAD_SIZE;
		    lookahead_count--;
		    hist[(hist_head+hist_count)%HIST_SIZE] = v;
		    hist_count += 1;

		  if (hist_count > HIST_SIZE) {
				hist_head = (hist_head + (1)) % HIST_SIZE;
				hist_count = HIST_SIZE;
		    }

		    used_bytes++;
		    

		}


		else {

	   		cur_len += (MIN_MATCH_LENGTH-1);

			uint32_t hist_start = hist_head+hist_count;
		    for (size_t i = 0; i < length; i++) {
				uint8_t z = lookahead[(lookahead_head+i) % LOOKAHEAD_SIZE];
				hist[(hist_start+i) % HIST_SIZE]= z;
		    }
		    
				    
			lookahead_head = (lookahead_head + length) % LOOKAHEAD_SIZE;
		    lookahead_count -= length;
		    hist_count += length;

		    if (hist_count > HIST_SIZE) {
				hist_head = (hist_head + (hist_count-HIST_SIZE)) % HIST_SIZE;
				hist_count = HIST_SIZE;
		    }

		    used_bytes += length;
			
		}


		if ((++blocks) == 8) {
			cur_len++;

		    blocks = 0;
		}

	
	  }

	  col_len[BLK_SIZE*chunk_idx + t_idx] = cur_len;
	  uint64_t cur_len_4B = ((cur_len + 3)/4)*4;
	  atomicAdd((unsigned long long int *)&block_len, (unsigned long long int )cur_len_4B);


	  __syncthreads();
	  if(threadIdx.x == 0){
	  	//128B alignment
	  	block_len = ((block_len + 127)/128)*128;
	  	blk_offset[chunk_idx+1] = (uint64_t)block_len;
	  }

}







__global__  void gpu_compress_func(const uint8_t* const in, uint8_t* out, const uint64_t in_n_bytes, const uint64_t out_n_bytes, 
					   					  const uint64_t in_chunk_size, const uint64_t out_chunk_size, const uint64_t n_chunks, 
					                      uint64_t* col_len,  uint8_t* col_map, uint64_t* blk_offset){

	uint8_t hist[HIST_SIZE] = {DEFAULT_CHAR};
    uint8_t lookahead[LOOKAHEAD_SIZE] = {DEFAULT_CHAR};

    uint32_t hist_head  = 0;
    uint32_t hist_count = 0;
    uint32_t lookahead_head = 0;
    uint32_t lookahead_count = 0;
    uint64_t consumed_bytes = 0;
    uint64_t out_bytes = 1;
    uint64_t cur_header_byte_pos = 0;
    uint8_t header_byte = 0;
    uint8_t blocks = 0;
    uint64_t used_bytes = 0;
    
    uint64_t in_off = 0;
	uint8_t in_flag = 0;

	int64_t my_chunk_size = (CHUNK_SIZE / BLK_SIZE);

	uint64_t byte_counter = 1;
	uint64_t byte_offset = 1;
	uint8_t col_counter = BLK_SIZE - 1;


    uint64_t in_start_idx;
    //change
	uint64_t chunk_offset = 0;

	int t_idx = threadIdx.x;
	int chunk_idx = blockIdx.x;

	uint8_t col_idx;

	for(int i = 0; i < 32;i++){
		if (t_idx ==  col_map[BLK_SIZE * chunk_idx + i]){
			col_idx = i;
		}
	}



	uint64_t out_start_idx = blk_offset[chunk_idx];


	in_start_idx = blockIdx.x * in_chunk_size;

	while(used_bytes < my_chunk_size){

		while ((lookahead_count < LOOKAHEAD_SIZE) && (consumed_bytes < my_chunk_size))  {
	
				uint8_t temp_v = in[in_start_idx + (consumed_bytes++) + in_off + t_idx*4];

				lookahead[(lookahead_head + (lookahead_count++)) % LOOKAHEAD_SIZE] = temp_v;			

				in_flag++;
				if(in_flag== 4){
					in_off += 31 * 4;
					in_flag = 0;
				}


			
		}




		uint32_t offset = 0;
		uint32_t length = 0;
		find_match(hist, hist_head, hist_count, lookahead, lookahead_head, lookahead_count, &offset, &length);
		if(t_idx == 0 && chunk_idx == 0 && used_bytes < 8){
			printf("length: %lu \n", length);
		}

		if (length == 0) {
		    uint8_t v = lookahead[lookahead_head];
   	//		out[out_start_idx + (out_bytes++) + byte_offset] = v;
		    out[out_start_idx + byte_offset + col_idx * 4] = v;

   			byte_counter++;
   			byte_offset++;
   	
   			if(byte_counter %4 ==0){
   				if(byte_counter >= (col_len[chunk_idx * BLK_SIZE + col_counter ]) ){
						col_counter--;

   				}
   				byte_offset += col_counter * 4;
   			}



		    lookahead_head = (lookahead_head + 1) % LOOKAHEAD_SIZE;
		    lookahead_count--;
		    hist[(hist_head+hist_count)%HIST_SIZE] = v;
		    hist_count += 1;

		  if (hist_count > HIST_SIZE) {
				hist_head = (hist_head + (1)) % HIST_SIZE;
				hist_count = HIST_SIZE;
		    }
		    header_byte = (header_byte | (1 << blocks));
		    used_bytes++;
		    //lens[t_idx].first++;

		}


		else {

	   		uint64_t v = (offset << LENGTH_SIZE) | (length - MIN_MATCH_LENGTH);

		    for (size_t i = 0; i < (MIN_MATCH_LENGTH-1); i++) {
				uint8_t k = v & 0x00FF;
				//out[out_start_idx + (out_bytes++)] = k;
				out[out_start_idx + byte_offset + col_idx * 4] = k;

				v >>= 8;

				byte_counter++;
	   			byte_offset++;
	   			//check

	   			if(byte_counter %4 ==0){
	   				if(byte_counter >= (col_len[chunk_idx * BLK_SIZE + col_counter]) ){
							col_counter--;

	   				}
	   				byte_offset += col_counter * 4;
	   			}

				//lens[t_idx].first++;
		    }
			uint32_t hist_start = hist_head+hist_count;
		    for (size_t i = 0; i < length; i++) {
				uint8_t z = lookahead[(lookahead_head+i) % LOOKAHEAD_SIZE];
				hist[(hist_start+i) % HIST_SIZE]= z;
		    }
		    
				    
			lookahead_head = (lookahead_head + length) % LOOKAHEAD_SIZE;
		    lookahead_count -= length;
		    hist_count += length;
		    if (hist_count > HIST_SIZE) {
				hist_head = (hist_head + (hist_count-HIST_SIZE)) % HIST_SIZE;
				hist_count = HIST_SIZE;
		    }

		    used_bytes += length;
			
		}

		//fix

		if ((++blocks) == 8) {

			if(out_start_idx + cur_header_byte_pos == 0){
				if(col_idx == 0){
					printf("header_byte: %x\n", header_byte );
					printf(": %u\n", threadIdx.x);
				}
			

			}

		    out[out_start_idx + cur_header_byte_pos + col_idx * 4] = header_byte;
		    header_byte = 0;
		    //cur_header_byte_pos = out_bytes++;
		    //lens[t_idx].first++;

		    cur_header_byte_pos = byte_offset;

		    byte_counter++;
		    byte_offset++;

		    if(byte_counter %4 ==0){
   				if(byte_counter >= (col_len[chunk_idx * BLK_SIZE + col_counter]) ){
						col_counter--;

   				}
   				byte_offset += col_counter * 4;
   			}


		    blocks = 0;
		}

		if (blocks != 0) {
			out[out_start_idx + cur_header_byte_pos + col_idx * 4] = header_byte;
	     }

	  }



}




    __host__ __device__ void compress_func(const uint8_t* const in, uint8_t* out, const uint64_t in_n_bytes, const uint64_t out_n_bytes, const uint64_t in_chunk_size, const uint64_t out_chunk_size, const uint64_t n_chunks, uint64_t* lens, const uint64_t tid) {
	if (tid < n_chunks) {
	    uint64_t rem = in_n_bytes % in_chunk_size;
	    uint64_t my_chunk_size = ((tid == (n_chunks - 1)) && (rem)) ? rem : in_chunk_size;
	    uint64_t in_start_idx = tid * in_chunk_size;
	    uint64_t out_start_idx = tid * out_chunk_size;

	    uint8_t hist[HIST_SIZE] = {DEFAULT_CHAR};
	    uint8_t lookahead[LOOKAHEAD_SIZE];
	    uint32_t hist_head  = 0;
	    uint32_t hist_count = 0;
	    uint32_t lookahead_head = 0;
	    uint32_t lookahead_count = 0;
	    uint64_t consumed_bytes = 0;
	    uint64_t out_bytes = 1;
	    uint64_t cur_header_byte_pos = 0;
	    uint8_t header_byte = 0;
	    uint8_t blocks = 0;
	    uint64_t c = 0;

	    uint64_t used_bytes = 0;

	    while (used_bytes < my_chunk_size) {
		
		//fill up lookahead buffer
		while ((lookahead_count < LOOKAHEAD_SIZE) && (consumed_bytes < my_chunk_size))  {
		    lookahead[(lookahead_head + (lookahead_count++)) % LOOKAHEAD_SIZE] =
			in[in_start_idx + (consumed_bytes++)];
		}
		//printf("Consumed: %llu\tChunk Size: %llu\n", (unsigned long long) consumed_bytes, (unsigned long long) my_chunk_size);
		uint32_t offset = 0;
		uint32_t length = 0;
		find_match(hist, hist_head, hist_count, lookahead, lookahead_head, lookahead_count, &offset, &length);
		//if (tid == 0 && c < 5)
		//	printf("HERE1: %c\t %llu\n", (char) lookahead[lookahead_head], (unsigned long long) c);
		//no match
		if (length == 0) {
		    uint8_t v = lookahead[lookahead_head];

		   

		    out[out_start_idx + (out_bytes++)] = v;
		    lookahead_head = (lookahead_head + 1) % LOOKAHEAD_SIZE;
		    lookahead_count--;
		    hist[(hist_head+hist_count)%HIST_SIZE] = v;
		    hist_count += 1;
		    if (hist_count > HIST_SIZE) {
			hist_head = (hist_head + (1)) % HIST_SIZE;
			hist_count = HIST_SIZE;
		    }
		    header_byte = (header_byte | (1 << blocks));

		    //if ((tid == 0) && (out_bytes < 100))
		    //    printf("b: %llu\t1: %c\t ub: %llu\n", (unsigned long long) out_bytes, (char)v, (unsigned long long) used_bytes);
		    used_bytes++;
		    
		    
		}
		//match
		else {
		    uint64_t v = (offset << LENGTH_SIZE) | (length - MIN_MATCH_LENGTH);
		    uint64_t v2 = v;

	

		    for (size_t i = 0; i < (MIN_MATCH_LENGTH-1); i++) {
			uint8_t k = v & 0x00FF;
			out[out_start_idx + (out_bytes++)] = k;
			//printf("k: %llu\n", (unsigned long long) k);
			v >>= 8;
		    }
		    uint32_t hist_start = hist_head+hist_count;
		    for (size_t i = 0; i < length; i++) {
			uint8_t z = lookahead[(lookahead_head+i) % LOOKAHEAD_SIZE];
			hist[(hist_start+i) % HIST_SIZE]= z;
			//if ((tid == 0) && (out_bytes < 100))
			//	printf("b: %llu\t1: %c\t ub: %llu\tleng: %llu\toffset: %llu\tj: %llu\tv: %p\n", (unsigned long long) out_bytes, (char)z, (unsigned long long) used_bytes, (unsigned long long) length, (unsigned long long) offset, (unsigned long long) i,v2);
		    }
		    
		    lookahead_head = (lookahead_head + length) % LOOKAHEAD_SIZE;
		    lookahead_count -= length;
		    hist_count += length;
		    if (hist_count > HIST_SIZE) {
			hist_head = (hist_head + (hist_count-HIST_SIZE)) % HIST_SIZE;
			hist_count = HIST_SIZE;
		    }
		    
		    //int offset_start = mod(hist_count - offset, HIST_SIZE);

		    
		    //header_byte = (header_byte << 1);
		    used_bytes += length;
		    
		}
		if ((++blocks) == 8) {
		    out[out_start_idx + cur_header_byte_pos] = header_byte;
		    header_byte = 0;
		    blocks = 0;
		    cur_header_byte_pos = out_bytes++;
		}
		//else
		//   header_byte <<= 1;
		c++;
	    }
	    if (blocks != 0) {
		out[out_start_idx + cur_header_byte_pos] = header_byte;

	    }
	    lens[tid] = out_bytes;
	    //if (out_bytes > my_chunk_size)
	    //printf("comrpessed larger than uncompressed\tout_bytes: %llu\tmy_chunk_size: %llu\n", (unsigned long long) out_bytes, (unsigned long long) my_chunk_size);
	    //printf("%llu done\n", (unsigned long long) tid);

	}
	
    }
    
    __host__ __device__ void compress_func_var_read(const uint8_t* const in, uint8_t* out, const uint64_t in_n_bytes, const uint64_t out_n_bytes, const uint64_t in_chunk_size, const uint64_t out_chunk_size, const uint64_t n_chunks, uint64_t* lens, const uint64_t tid) {
	if (tid < n_chunks) {
	    uint64_t rem = in_n_bytes % in_chunk_size;
	    uint64_t my_chunk_size = ((tid == (n_chunks - 1)) && (rem)) ? rem : in_chunk_size;
	    uint64_t in_start_idx = tid * in_chunk_size;
	    uint64_t out_start_idx = tid * out_chunk_size;

	    uint8_t hist[HIST_SIZE] = {DEFAULT_CHAR};
	    uint8_t lookahead[LOOKAHEAD_SIZE];
	    uint32_t hist_head  = 0;
	    uint32_t hist_count = 0;
	    uint32_t lookahead_head = 0;
	    uint32_t lookahead_count = 0;
	    uint64_t consumed_bytes = 0;
	    uint64_t out_bytes = 1;
	    uint64_t cur_header_byte_pos = 0;
	    uint8_t header_byte = 0;
	    uint8_t blocks = 0;
	    uint64_t c = 0;

	    uint64_t used_bytes = 0;
	    uint32_t lookahead_head_4 = 0;
	    uint32_t lookahead_count_4 =0;
	    const uint32_t* const in_4 = (const uint32_t*) (in + in_start_idx);
	    uint32_t* lookahead_4 = ( uint32_t*) lookahead;
	    uint32_t consumed_bytes_4 = 0;
	    uint32_t lookahead_head_mod = 0;
	    while (used_bytes < my_chunk_size) {
		
		//fill up lookahead buffer
		while ((lookahead_count_4 < LOOKAHEAD_UNITS) && (consumed_bytes < my_chunk_size))  {
		    lookahead_4[(lookahead_head_4 + (lookahead_count_4++)) % LOOKAHEAD_UNITS] =
			in_4[(consumed_bytes_4++)];
		    consumed_bytes += READ_UNITS;
		    lookahead_count += READ_UNITS;
		    
		    if (consumed_bytes > my_chunk_size) {
			uint32_t diff = consumed_bytes - my_chunk_size;
			consumed_bytes -= diff;
			lookahead_count -= diff;
			break;
		    }
		}
		//printf("Consumed: %llu\tChunk Size: %llu\n", (unsigned long long) consumed_bytes, (unsigned long long) my_chunk_size);
		uint32_t offset = 0;
		uint32_t length = 0;
		find_match(hist, hist_head, hist_count, lookahead, lookahead_head, lookahead_count, &offset, &length);
		//if (tid == 0 && c < 5)
		//	printf("HERE1: %c\t %llu\n", (char) lookahead[lookahead_head], (unsigned long long) c);
		//no match
		if (length == 0) {
		    uint8_t v = lookahead[lookahead_head];
		    out[out_start_idx + (out_bytes++)] = v;
		    lookahead_head = (lookahead_head + 1) % LOOKAHEAD_SIZE;
		    lookahead_count--;
		    hist[(hist_head+hist_count)%HIST_SIZE] = v;
		    hist_count += 1;
		    if (hist_count > HIST_SIZE) {
			hist_head = (hist_head + (1)) % HIST_SIZE;
			hist_count = HIST_SIZE;
		    }
		    header_byte = (header_byte | (1 << blocks));

		    //if ((tid == 0) && (out_bytes < 100))
		    //    printf("b: %llu\t1: %c\t ub: %llu\n", (unsigned long long) out_bytes, (char)v, (unsigned long long) used_bytes);
		    used_bytes++;
		    lookahead_head_mod++;
		    if (lookahead_head_mod == READ_UNITS) {
			lookahead_head_mod = 0;
			lookahead_head_4 = (lookahead_head_4 + 1) % LOOKAHEAD_UNITS;
			lookahead_count_4--;
			
		    }
		    
		}
		//match
		else {
		    uint64_t v = (offset << LENGTH_SIZE) | (length - MIN_MATCH_LENGTH);
		    uint64_t v2 = v;
		    for (size_t i = 0; i < (MIN_MATCH_LENGTH-1); i++) {
			uint8_t k = v & 0x00FF;
			out[out_start_idx + (out_bytes++)] = k;
			//printf("k: %llu\n", (unsigned long long) k);
			v >>= 8;
		    }
		    uint32_t hist_start = hist_head+hist_count;
		    for (size_t i = 0; i < length; i++) {
			uint8_t z = lookahead[(lookahead_head+i) % LOOKAHEAD_SIZE];
			hist[(hist_start+i) % HIST_SIZE]= z;
			//if ((tid == 0) && (out_bytes < 100))
			//	printf("b: %llu\t1: %c\t ub: %llu\tleng: %llu\toffset: %llu\tj: %llu\tv: %p\n", (unsigned long long) out_bytes, (char)z, (unsigned long long) used_bytes, (unsigned long long) length, (unsigned long long) offset, (unsigned long long) i,v2);
		    }
		    
		    lookahead_head = (lookahead_head + length) % LOOKAHEAD_SIZE;
		    lookahead_count -= length;
		    hist_count += length;
		    if (hist_count > HIST_SIZE) {
			hist_head = (hist_head + (hist_count-HIST_SIZE)) % HIST_SIZE;
			hist_count = HIST_SIZE;
		    }
		    
		    //int offset_start = mod(hist_count - offset, HIST_SIZE);

		    
		    //header_byte = (header_byte << 1);
		    used_bytes += length;
		    lookahead_head_mod+=length;
		    if (lookahead_head_mod >= READ_UNITS) {
			uint32_t k = lookahead_head_mod / READ_UNITS;
			lookahead_head_mod = lookahead_head_mod % READ_UNITS;
			lookahead_head_4 = (lookahead_head_4 + k) % LOOKAHEAD_UNITS;
			lookahead_count_4-=k;
			
		    }
		}
		if ((++blocks) == 8) {
		    out[out_start_idx + cur_header_byte_pos] = header_byte;
		    header_byte = 0;
		    blocks = 0;
		    cur_header_byte_pos = out_bytes++;
		}
		//else
		//   header_byte <<= 1;
		c++;
	    }
	    if (blocks != 0) {
		out[out_start_idx + cur_header_byte_pos] = header_byte;

	    }
	    lens[tid] = out_bytes;
	    //if (out_bytes > my_chunk_size)
	    //printf("comrpessed larger than uncompressed\tout_bytes: %llu\tmy_chunk_size: %llu\n", (unsigned long long) out_bytes, (unsigned long long) my_chunk_size);
	    //printf("%llu done\n", (unsigned long long) tid);

	}
	
    }
    __device__ void compress_func_proto(const uint8_t* const in, uint8_t* out, const uint64_t in_n_bytes, const uint64_t out_n_bytes, const uint64_t in_chunk_size, const uint64_t out_chunk_size, const uint64_t n_chunks, uint64_t* lens, const uint64_t tid) {
	uint64_t warp_id = WARP_ID(tid);
	uint64_t lane_id = tid%32;
	typedef cub::WarpReduce<uint64_t> WarpReduce;
	__shared__ typename WarpReduce::TempStorage temp_storage[BLK_SIZE/32];
	__shared__ uint8_t warp_mappings[32];
	__shared__ uint32_t warp_lens[32];
	__shared__ uint64_t s_aggregate;
	if (warp_id < n_chunks) {
	    unsigned active  = __activemask();
	    uint8_t out_[OVERHEAD_PER_CHUNK(CHUNK_SIZE/32)];
	    uint64_t rem = in_n_bytes % in_chunk_size;
	    uint64_t our_chunk_size = ((warp_id == (n_chunks - 1)) && (rem)) ? rem : in_chunk_size;
	    uint64_t my_chunk_size = our_chunk_size/32;
	    uint64_t in_start_idx = warp_id * in_chunk_size;
	    uint64_t out_start_idx = warp_id * out_chunk_size;

	    uint8_t hist[HIST_SIZE] = {DEFAULT_CHAR};
	    uint8_t lookahead[LOOKAHEAD_SIZE];
	    uint32_t hist_head  = 0;
	    uint32_t hist_count = 0;
	    uint32_t lookahead_head = 0;
	    uint32_t lookahead_count = 0;
	    uint64_t consumed_bytes = 0;
	    uint64_t out_bytes = 1;
	    uint64_t cur_header_byte_pos = 0;
	    uint8_t header_byte = 0;
	    uint8_t blocks = 0;
	    uint64_t c = 0;

	    uint64_t used_bytes = 0;
	    uint32_t lookahead_head_4 = 0;
	    uint32_t lookahead_count_4 =0;
	    const uint32_t* const in_4 = (const uint32_t*) (in + in_start_idx);
	    uint32_t* lookahead_4 = ( uint32_t*) lookahead;
	    uint32_t consumed_bytes_4 = 0;
	    uint32_t lookahead_head_mod = 0;
	    uint32_t iter = 0;
	    while (used_bytes < my_chunk_size) {
		unsigned active2 = __activemask();
		uint64_t min = WarpReduce(temp_storage[warp_id]).Reduce(LOOKAHEAD_UNITS-lookahead_count_4, cub::Min());
		uint64_t r = lookahead_count_4 + min;
		min = __shfl_sync(active2, min, 0);
		//fill up lookahead buffer
		while ((lookahead_count_4 < r) && (consumed_bytes < my_chunk_size))  {
		    lookahead_4[(lookahead_head_4 + (lookahead_count_4++)) % LOOKAHEAD_UNITS] =
			in_4[(consumed_bytes_4++)*32 + lane_id];
		    consumed_bytes += READ_UNITS;
		    lookahead_count += READ_UNITS;
		    
		    if (consumed_bytes > my_chunk_size) {
			uint32_t diff = consumed_bytes - my_chunk_size;
			consumed_bytes -= diff;
			lookahead_count -= diff;
			//break;
		    }
		}
		//printf("Consumed: %llu\tChunk Size: %llu\n", (unsigned long long) consumed_bytes, (unsigned long long) my_chunk_size);
		uint32_t offset = 0;
		uint32_t length = 0;
		find_match(hist, hist_head, hist_count, lookahead, lookahead_head, lookahead_count, &offset, &length);
		//if (tid == 0 && c < 5)
		//	printf("HERE1: %c\t %llu\n", (char) lookahead[lookahead_head], (unsigned long long) c);
		//no match
		if (length == 0) {
		    uint8_t v = lookahead[lookahead_head];
		    out_[ (out_bytes++)] = v;
		    lookahead_head = (lookahead_head + 1) % LOOKAHEAD_SIZE;
		    lookahead_count--;
		    hist[(hist_head+hist_count)%HIST_SIZE] = v;
		    hist_count += 1;
		    if (hist_count > HIST_SIZE) {
			hist_head = (hist_head + (1)) % HIST_SIZE;
			hist_count = HIST_SIZE;
		    }
		    header_byte = (header_byte | (1 << blocks));

		    //if ((tid == 0) && (out_bytes < 100))
		    //    printf("b: %llu\t1: %c\t ub: %llu\n", (unsigned long long) out_bytes, (char)v, (unsigned long long) used_bytes);
		    used_bytes++;
		    lookahead_head_mod++;
		    if (lookahead_head_mod == READ_UNITS) {
			lookahead_head_mod = 0;
			lookahead_head_4 = (lookahead_head_4 + 1) % LOOKAHEAD_UNITS;
			lookahead_count_4--;
			
		    }
		    
		}
		//match
		else {
		    uint64_t v = (offset << LENGTH_SIZE) | (length - MIN_MATCH_LENGTH);
		    uint64_t v2 = v;
		    for (size_t i = 0; i < (MIN_MATCH_LENGTH-1); i++) {
			uint8_t k = v & 0x00FF;
			out_[(out_bytes++)] = k;
			//printf("k: %llu\n", (unsigned long long) k);
			v >>= 8;
		    }
		    uint32_t hist_start = hist_head+hist_count;
		    for (size_t i = 0; i < length; i++) {
			uint8_t z = lookahead[(lookahead_head+i) % LOOKAHEAD_SIZE];
			hist[(hist_start+i) % HIST_SIZE]= z;
			//if ((tid == 0) && (out_bytes < 100))
			//	printf("b: %llu\t1: %c\t ub: %llu\tleng: %llu\toffset: %llu\tj: %llu\tv: %p\n", (unsigned long long) out_bytes, (char)z, (unsigned long long) used_bytes, (unsigned long long) length, (unsigned long long) offset, (unsigned long long) i,v2);
		    }
		    
		    lookahead_head = (lookahead_head + length) % LOOKAHEAD_SIZE;
		    lookahead_count -= length;
		    hist_count += length;
		    if (hist_count > HIST_SIZE) {
			hist_head = (hist_head + (hist_count-HIST_SIZE)) % HIST_SIZE;
			hist_count = HIST_SIZE;
		    }
		    
		    //int offset_start = mod(hist_count - offset, HIST_SIZE);

		    
		    //header_byte = (header_byte << 1);
		    used_bytes += length;
		    lookahead_head_mod+=length;
		    if (lookahead_head_mod >= READ_UNITS) {
			uint32_t k = lookahead_head_mod / READ_UNITS;
			lookahead_head_mod = lookahead_head_mod % READ_UNITS;
			lookahead_head_4 = (lookahead_head_4 + k) % LOOKAHEAD_UNITS;
			lookahead_count_4-=k;
			
		    }
		}
		if ((++blocks) == 8) {
		    out_[cur_header_byte_pos] = header_byte;
		    header_byte = 0;
		    blocks = 0;
		    cur_header_byte_pos = out_bytes++;
		}
		//else
		//   header_byte <<= 1;
		c++;
	    }
	    if (blocks != 0) {
		out_[cur_header_byte_pos] = header_byte;

	    }
	    warp_lens[lane_id] = out_bytes;
	    __syncthreads();
	    uint64_t aggregate = 0;
	    if (lane_id == 0) {
		for (size_t i = 0; i < 32; i++) {
		    uint32_t smaller_than_me = 0;
		    uint32_t equal_to_me = 0;
		    for (size_t j = 0; j< 32; j++) {
			if (i != j) {
			    if (warp_lens[i] > warp_lens[j])
				smaller_than_me++;
			    else if ((j < i) && (warp_lens[i] == warp_lens[j]))
				equal_to_me++;
			}
		    }
		    aggregate += ceil<uint32_t>(warp_lens[i], READ_UNITS)+1;
		    warp_mappings[i] = smaller_than_me + equal_to_me;
		    
		}
		s_aggregate = aggregate;

	    }
	    __syncthreads();
	    uint32_t* out_4_ = (uint32_t*) (out + out_start_idx);
	    uint32_t* out_4_in = (uint32_t*) out_;
	    uint32_t my_map = warp_mappings[lane_id];
	    uint32_t n_out_4 = ceil<uint32_t>(out_bytes, READ_UNITS)+1;
	    out_4_[my_map] = n_out_4;
	    uint32_t out_4 = 1;
	    uint32_t out_4__ = 32* out_4;
	    while (out_4 < n_out_4) {
		unsigned active_ = __activemask();
		unsigned c_a = __popc(active_);
		out_4_[out_4__ + my_map] = out_4_in[out_4-1];
		out_4__ += c_a;
		out_4++;
	    }
	    if (lane_id == 0)
		lens[warp_id] = s_aggregate;
	    //if (out_bytes > my_chunk_size)
	    //printf("comrpessed larger than uncompressed\tout_bytes: %llu\tmy_chunk_size: %llu\n", (unsigned long long) out_bytes, (unsigned long long) my_chunk_size);
	    //printf("%llu done\n", (unsigned long long) tid);

	}
	
    }

	__global__ void
    __launch_bounds__(32, 32)
	kernel_decompress(const uint8_t* const in, uint8_t* out, const uint64_t in_n_bytes, const uint64_t out_n_bytes, const uint64_t out_chunk_size, const uint64_t n_chunks, const uint64_t* const blk_off, const uint64_t* const col_len, const uint8_t* const col_map) {
	uint64_t tid = threadIdx.x + blockDim.x * blockIdx.x;
	decompress_func_new(in, out, in_n_bytes, out_n_bytes, out_chunk_size, n_chunks, blk_off, col_len, col_map);
    }
    __global__ void
    __launch_bounds__(32, 32)
    kernel_decompress(const uint8_t* const in, uint8_t* out, const uint64_t in_n_bytes, const uint64_t out_n_bytes, const uint64_t out_chunk_size, const uint64_t n_chunks, const uint64_t* const lens) {
	uint64_t tid = threadIdx.x + blockDim.x * blockIdx.x;
	decompress_func(in, out, in_n_bytes, out_n_bytes, out_chunk_size, n_chunks, lens, tid);
    }

    __global__ void
    __launch_bounds__(32, 32)
    kernel_compress(const uint8_t* const in, uint8_t* out, const uint64_t in_n_bytes, const uint64_t out_n_bytes, const uint64_t in_chunk_size, const uint64_t out_chunk_size, const uint64_t n_chunks, uint64_t* lens) {
	uint64_t tid = threadIdx.x + blockDim.x * blockIdx.x;
	compress_func_var_read(in, out, in_n_bytes, out_n_bytes, in_chunk_size, out_chunk_size, n_chunks, lens, tid);
    }

    __host__ __device__ void shift_data_func(const uint8_t* const in, uint8_t* out, const uint64_t* const lens, const uint64_t in_chunk_size, const uint64_t n_chunks, uint64_t tid) {
	if (tid < n_chunks) {
	    
	    uint64_t out_start = (tid == 0) ? 0 : (lens[tid-1]);
	    uint64_t out_end = lens[tid];
	    const uint8_t* const in_start = in + (tid*in_chunk_size);
	    uint64_t n = out_end - (out_start);
	    if (tid == (n_chunks -1))
		printf("out_start: %llu\tout_end: %llu\tn: %llu\n", (unsigned long long) out_start, (unsigned long long) out_end, (unsigned long long) n);
	    for (size_t i = 0; i < n; i++) {
		out[out_start+i] = in_start[i];
	    }

	}
    }

    __global__ void
    __launch_bounds__(1024, 2)
    kernel_shift_data(const uint8_t* const in, uint8_t* out, const uint64_t* const lens, const uint64_t in_chunk_size, const uint64_t n_chunks) {
	uint64_t tid = threadIdx.x + blockDim.x * blockIdx.x;
	shift_data_func(in, out, lens, in_chunk_size, n_chunks, tid);
    }

    __host__ void compress_gpu(const uint8_t* const in, uint8_t** out, const uint64_t in_n_bytes, uint64_t* out_n_bytes) {
	uint8_t* d_in;
	uint8_t* d_out;
	uint8_t* temp;

	uint64_t padded_in_n_bytes = in_n_bytes;// + (CHUNK_SIZE-(in_n_bytes % CHUNK_SIZE));
	uint32_t n_chunks = padded_in_n_bytes/CHUNK_SIZE;
	uint32_t chunk_size = padded_in_n_bytes/n_chunks;
	assert((chunk_size % READ_UNITS)==0);
	uint64_t exp_out_chunk_size = (chunk_size+OVERHEAD_PER_CHUNK_(chunk_size));
	uint64_t exp_data_out_bytes = (n_chunks*exp_out_chunk_size);
	uint64_t len_bytes =  (n_chunks*sizeof(uint64_t));
	uint64_t head_bytes = HEAD_INTS*sizeof(uint32_t);
	uint64_t out_bytes = head_bytes +  //header
	    len_bytes +  //lens
	    exp_data_out_bytes; //data

/*

	cuda_err_chk(cudaMalloc(&d_in, padded_in_n_bytes));
	cuda_err_chk(cudaMalloc(&d_out, len_bytes + exp_data_out_bytes));
	uint64_t* d_lens_out = (uint64_t*) d_out;
	uint8_t* d_data_out = d_out + len_bytes;
	printf("padded_bytes: %llu\tLENGTH_MASK: %p\texpected_out_bytes: %llu\toverhead_per_chunk: %llu\tchunk_size: %llu\n", padded_in_n_bytes, LENGTH_MASK(LENGTH_SIZE), exp_data_out_bytes, OVERHEAD_PER_CHUNK_(chunk_size), chunk_size);
	//return;
	cuda_err_chk(cudaMemcpy(d_in, in, in_n_bytes, cudaMemcpyHostToDevice));
	uint64_t grid_size = ceil<uint64_t>(n_chunks, BLK_SIZE);
    kernel_compress<<<grid_size, BLK_SIZE>>>(d_in, d_data_out, in_n_bytes, exp_data_out_bytes, chunk_size, exp_out_chunk_size, n_chunks, d_lens_out);
	cuda_err_chk(cudaDeviceSynchronize());
	printf("Compress Kernel Done\n");
	thrust::inclusive_scan(thrust::device, d_lens_out, d_lens_out + n_chunks, d_lens_out);
	cuda_err_chk(cudaDeviceSynchronize());
	cuda_err_chk(cudaFree(d_in));
	cuda_err_chk(cudaMalloc(&temp, exp_data_out_bytes));

	kernel_shift_data<<<grid_size, BLK_SIZE>>>(d_data_out, temp, d_lens_out, exp_out_chunk_size, n_chunks);
	cuda_err_chk(cudaDeviceSynchronize());

*/
	printf("in bytes: %llu\n", in_n_bytes);


	uint64_t num_chunk = in_n_bytes / CHUNK_SIZE;
	printf("cpu num chunk: %llu\n", num_chunk);



	//cpu
	uint8_t* cpu_data_out = (uint8_t*) malloc(exp_data_out_bytes);
	uint64_t* len_out = (uint64_t*) malloc(sizeof(uint64_t) * BLK_SIZE * num_chunk);
	uint8_t* col_map = (uint8_t*) malloc(BLK_SIZE * num_chunk);
	uint64_t* blk_offset = (uint64_t*) malloc(8*(num_chunk + 1));
	uint64_t* chunk_offset = (uint64_t*) malloc(8*(num_chunk + 1));
	uint64_t* col_offset = (uint64_t*) malloc(8*(BLK_SIZE * num_chunk + 1));

	cpu_compress_func(in, cpu_data_out, in_n_bytes, exp_data_out_bytes, chunk_size, exp_out_chunk_size, n_chunks, len_out, col_map, blk_offset, col_offset, chunk_offset);




	uint64_t* d_blk_offset;
	uint64_t* d_col_len;
	uint8_t*  d_col_map;

	cuda_err_chk(cudaMalloc(&d_in, padded_in_n_bytes));
	cuda_err_chk(cudaMalloc(&d_col_len, sizeof(uint64_t) * BLK_SIZE * num_chunk));
	cuda_err_chk(cudaMalloc(&d_col_map, BLK_SIZE * num_chunk));
	cuda_err_chk(cudaMalloc(&d_blk_offset, 8*(num_chunk + 1)));




	cuda_err_chk(cudaMemcpy(d_in, in, in_n_bytes, cudaMemcpyHostToDevice));

	gpu_compress_init<<<n_chunks, BLK_SIZE>>>(d_in, in_n_bytes, 
											  chunk_size,exp_out_chunk_size, n_chunks,
											   d_col_len,  d_blk_offset);
	cuda_err_chk(cudaDeviceSynchronize()); 

	// ExampleKernel<<<n_chunks, BLK_SIZE>>>(d_col_len, d_col_map);
	parallel_scan<<<1,1>>>(d_blk_offset, n_chunks);
	cuda_err_chk(cudaDeviceSynchronize()); 

	// cuda_err_chk(cudaMemcpy(len_out, d_col_len, sizeof(uint64_t) * BLK_SIZE * num_chunk, cudaMemcpyDeviceToHost));
	cuda_err_chk(cudaMemcpy(blk_offset, d_blk_offset,  8*(num_chunk + 1), cudaMemcpyDeviceToHost));
	//cuda_err_chk(cudaMemcpy(col_map, d_col_map, BLK_SIZE * num_chunk, cudaMemcpyDeviceToHost));



	cuda_err_chk(cudaMemcpy(d_col_map, col_map, BLK_SIZE * num_chunk, cudaMemcpyHostToDevice));
	//cuda_err_chk(cudaMemcpy(d_blk_offset, blk_offset,  8*(num_chunk + 1), cudaMemcpyHostToDevice));
	cuda_err_chk(cudaMemcpy(d_col_len, len_out, sizeof(uint64_t) * BLK_SIZE * num_chunk, cudaMemcpyHostToDevice));



	uint64_t final_out_size = blk_offset[num_chunk];
	cuda_err_chk(cudaMalloc(&d_out, final_out_size));


	
	*out = new uint8_t[final_out_size];
	printf("malloc out\n");


	


	gpu_compress_func<<<n_chunks, BLK_SIZE>>>(d_in, d_out, in_n_bytes, exp_data_out_bytes, 
											  chunk_size,exp_out_chunk_size, n_chunks,
											   d_col_len, d_col_map, d_blk_offset);

	

	uint64_t out_data_bytes = blk_offset[num_chunk];
	printf("out_data_bytes: %llu\n", out_data_bytes);

	std::ofstream col_map_file ("col_map.bin",std::ofstream::binary);
	col_map_file.write ((const char *)(col_map),  BLK_SIZE * num_chunk);
	col_map_file.close();

	std::ofstream col_len_file ("col_len.bin",std::ofstream::binary);
	col_len_file.write ((const  char *)(len_out),  BLK_SIZE * num_chunk * 8);
	col_len_file.close();


	std::ofstream blk_off_file ("blk_offset.bin",std::ofstream::binary);
	blk_off_file.write ((const char *)(blk_offset), (num_chunk + 1)*8);
	blk_off_file.close();




	cuda_err_chk(cudaMemcpy((*out), d_out, out_data_bytes, cudaMemcpyDeviceToHost));

	//*out_n_bytes = head_bytes + len_bytes + out_data_bytes;

	*out_n_bytes = out_data_bytes;
	
	printf("first byte: %p\n", ((*out) )[0]);

	cuda_err_chk(cudaFree(d_out));
	cuda_err_chk(cudaFree(temp));
	printf("done");	
    }


 __host__ void decompress_gpu(const uint8_t* const in, uint8_t** out, const uint64_t in_n_bytes, uint64_t* out_n_bytes) {
	uint8_t* d_in;
	uint8_t* d_out;
	uint64_t* d_lens;

	const uint64_t* const in_64 = (const uint64_t*) in;


	int blk_off_fd;
	struct stat blk_off_sb;
	uint64_t* blk_off;
	if((blk_off_fd = open("blk_offset.bin", O_RDONLY)) == 0) {
	    printf("Fatal Error: blk_offset File open error\n");
	    //return -1;
	}

	fstat(blk_off_fd, &blk_off_sb);

	blk_off = (uint64_t*) mmap(nullptr, blk_off_sb.st_size, PROT_READ, MAP_PRIVATE, blk_off_fd, 0);

	if(blk_off == (void*)-1){
	    printf("Fatal Error: blk_offset Mapping error\n");
	    //return -1;
	}

	int col_len_fd;
	struct stat col_len_sb;
	uint64_t* col_len;
	if((col_len_fd = open("col_len.bin", O_RDONLY)) == 0) {
	    printf("Fatal Error: col_len File open error\n");
	    //return -1;
	}

	fstat(col_len_fd, &col_len_sb);

	col_len = (uint64_t*)mmap(nullptr, col_len_sb.st_size, PROT_READ, MAP_PRIVATE, col_len_fd, 0);

	if(col_len == (void*)-1){
	    printf("Fatal Error: col_len Mapping error\n");
	    //return -1;
	}

	int col_map_fd;
	struct stat col_map_sb;
	uint8_t* col_map;
	if((col_map_fd = open("col_map.bin", O_RDONLY)) == 0) {
	    printf("Fatal Error: col_map File open error\n");
	    //return -1;
	}

	fstat(col_map_fd, &col_map_sb);

	col_map = (uint8_t*)mmap(nullptr, col_map_sb.st_size, PROT_READ, MAP_PRIVATE, col_map_fd, 0);

	if(col_map == (void*)-1){
	    printf("Fatal Error: col_map Mapping error\n");
	    //return -1;
	}

	uint64_t n_chunks = (blk_off_sb.st_size/sizeof(uint64_t)) - 1;
	uint64_t out_size = n_chunks * CHUNK_SIZE;


	uint64_t* d_blk_off;
	cuda_err_chk(cudaMalloc(&d_blk_off, blk_off_sb.st_size));

	uint64_t* d_col_len;
	cuda_err_chk(cudaMalloc(&d_col_len, col_len_sb.st_size));

	uint8_t* d_col_map;
	cuda_err_chk(cudaMalloc(&d_col_map, col_map_sb.st_size));


	cuda_err_chk(cudaMemcpy(d_blk_off, blk_off, blk_off_sb.st_size, cudaMemcpyHostToDevice));

	cuda_err_chk(cudaMemcpy(d_col_len, col_len, col_len_sb.st_size, cudaMemcpyHostToDevice));

	cuda_err_chk(cudaMemcpy(d_col_map, col_map, col_map_sb.st_size, cudaMemcpyHostToDevice));




	*out_n_bytes = out_size;
	uint64_t out_bytes = *out_n_bytes;

	/*
	const uint32_t* const in_32 = (const uint32_t*) (in + sizeof(uint64_t));
	uint32_t chunk_size = in_32[0];
	uint32_t n_chunks = in_32[1];
	uint32_t hist_size = in_32[2];
	uint32_t min_match_len = in_32[3];
	uint32_t max_match_len = in_32[4];


	uint64_t len_bytes =  (n_chunks*sizeof(uint64_t));
	uint64_t head_bytes = HEAD_INTS*sizeof(uint32_t);
	*/
	//const uint64_t* const lens = (const uint64_t*) (in + head_bytes);
	const uint8_t* const in_ = in ;


	cuda_err_chk(cudaMalloc(&d_in, in_n_bytes));
	cuda_err_chk(cudaMalloc(&d_out, (*out_n_bytes)*2));
	//printf("out_bytes: %p\td_out: %p\tchunk_size: %llu\tn_chunks: %llu\tHIST_SIZE: %llu\tMIN_MATCH_LENGTH: %llu\tMAX_MATCH_LENGTH: %llu\tOFFSET_SIZE: %llu\tbitsNeeded(4096): %llu\n",
	//out_bytes, d_out, chunk_size, n_chunks, hist_size, min_match_len, max_match_len, OFFSET_SIZE, bitsNeeded(HIST_SIZE));

	//return;

	cuda_err_chk(cudaMemcpy(d_in, in_, in_n_bytes, cudaMemcpyHostToDevice));


	dim3 grid_size( n_chunks);
	dim3 blk(BLK_SIZE,1);
	std::chrono::high_resolution_clock::time_point kernel_start = std::chrono::high_resolution_clock::now();
        kernel_decompress<<<grid_size, blk>>>(d_in, d_out, in_n_bytes, out_size, CHUNK_SIZE, n_chunks, d_blk_off, d_col_len, d_col_map);
	cuda_err_chk(cudaDeviceSynchronize());
	std::chrono::high_resolution_clock::time_point kernel_end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> total = std::chrono::duration_cast<std::chrono::duration<double>>(kernel_end - kernel_start);
	std::cout << "kernel time: " << total.count() << " secs\n";
	printf("in bytes: %llu\n", in_n_bytes);

	*out = new uint8_t[out_bytes];
	cuda_err_chk(cudaMemcpy((*out), d_out, out_bytes, cudaMemcpyDeviceToHost));
	printf("bytes: %llu\n",*out_n_bytes );


	cuda_err_chk(cudaFree(d_out));
	cuda_err_chk(cudaFree(d_in));

    }




}
