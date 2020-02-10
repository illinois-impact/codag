
#include <common.h>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <fstream>

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




void cpu_shift(const uint8_t* in, uint8_t* out, uint64_t* blk_offset, const uint64_t* len_out, const uint8_t* col_map , const uint64_t* col_offset, uint64_t* chunk_offset, uint64_t num_chunk){


	uint64_t cur_copied[BLK_SIZE] = {0};
	uint64_t byte_read[BLK_SIZE] = {0};
	uint64_t out_bytes = 0;



	for(int chunk_id = 0; chunk_id < num_chunk; chunk_id++){


		if(chunk_id % 100 == 0){
			printf("chunk_id: %i\n", chunk_id);
		}


		for (int j = 0; j < BLK_SIZE; j++){
			cur_copied[j] = 0;
			byte_read[j] = 0;
		}
		uint64_t b_offset = blk_offset[chunk_id];
		uint64_t chunk_len = chunk_offset[chunk_id+1] - chunk_offset[chunk_id];
		uint64_t copied_len = 0;

		// cur_copied[BLK_SIZE] = {0};
		// byte_read[BLK_SIZE] = {0};

		//printf("chunk len: %llu\n",chunk_len );

		while(copied_len < chunk_len){


			//printf("copied_len %llu\n",copied_len );

			for(int t_idx = 0; t_idx < BLK_SIZE; t_idx++){

				uint8_t col_idx = col_map[chunk_id * BLK_SIZE + t_idx] + BLK_SIZE * chunk_id;
				//uint8_t col_idx = col_map[chunk_id * BLK_SIZE + t_idx];


				//uint64_t col_len = col_offset[BLK_SIZE * chunk_id + col_idx + 1] - col_offset[BLK_SIZE * chunk_id + col_idx];
				//uint64_t col_len = col_offset[col_idx + 1] - col_offset[col_idx];
				uint64_t col_len = len_out[t_idx + BLK_SIZE*chunk_id];

				if(cur_copied[t_idx] < col_len){
					for(int i = 0; i < 4; i++){
						//printf("%x", in[b_offset + col_offset[BLK_SIZE * chunk_id + col_idx] + byte_read[t_idx]]);

						//4B padding
						if(cur_copied[t_idx] == col_len){
							out[out_bytes] = 0;
							copied_len++;
							out_bytes++;
						}
						else{
							// out[out_bytes] = in[b_offset + col_offset[BLK_SIZE * chunk_id + col_idx] + byte_read[t_idx]];
							out[out_bytes] = in[b_offset + col_offset[col_idx] + byte_read[t_idx]];

							byte_read[t_idx]++;
							cur_copied[t_idx]++;
							copied_len++;
							out_bytes++;

						}

		
					}
				}
			}

		}


		// uint8_t rem = 128 - out_bytes % 128;
		// if(rem != 128){
		// 			out_bytes += rem;

		// }

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
		out_start_idx = chunk_offset;


		

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
		//check the constant
		//chunk_offset += ceil(chunk_len / 128) * 128;
		chunk_offset += chunk_len;
		chunk_offset_array[chunk_id+1] = chunk_offset;
		//chunk_offset += chunk_len;

	
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

	printf("cpu compress func done\n");


	
	// printf("len_out\n");
	// for(int i = 0; i < num_chunk * BLK_SIZE; i++){
	// 	printf("%llu\n", len_out[i]);
	// }


	// printf("col_map\n");
	// for(int i = 0; i < num_chunk * BLK_SIZE; i++){
	// 	printf("%llu\n", col_map[i]);
	// }

	// printf("blk_offset\n");
	// for(int i = 0; i < num_chunk + 1; i++){
	// 	printf("%llu\n", blk_offset[i]);
	// }

	


	// printf("col_offset\n");
	// for(int i = 0; i < num_chunk * BLK_SIZE + 1; i++){
	// 	printf("%llu\n", col_offset[i]);
	// }



	// uint64_t four_padding = 0;

	// for(int i = 0; i < num_chunk * BLK_SIZE; i++){
	// 	uint8_t rem = len_out[i] % 4;
	// 	if(rem != 0){
	// 		four_padding += 4 - rem;
	// 	}
	// }


	uint64_t four_padding = 0;
	for(int i = 0; i < num_chunk; i++){
		for(int j = 0; j < BLK_SIZE; j++){

			uint8_t rem = len_out[i*BLK_SIZE + j] % 4;
				if(rem != 0){
					four_padding += 4 - rem;
				}
		}
		chunk_offset[i+1] += four_padding; 
	}

// printf("chunk offset\n");
// 	for(int i = 0; i < num_chunk + 1; i++){
// 		printf("%llu\n", chunk_offset[i]);
// 	}




	//uint64_t final_out_size = four_padding + chunk_offset[num_chunk];
	
	uint64_t final_out_size = chunk_offset[num_chunk];


	uint64_t blk_padding = 0;
	for(int i = 0; i < num_chunk; i++){
		blk_offset[i+1] = chunk_offset[i+1]; 
		uint64_t temp_len = chunk_offset[i+1] - chunk_offset[i];
		uint8_t rem = temp_len %128;
		if(rem != 0){
			blk_padding += 128 - rem;
			blk_offset[i+1] += blk_padding;
		}
	}

	final_out_size += blk_padding;


	printf("setting final output size\n");
	printf("final out size: %llu\n", final_out_size);

	uint8_t* cpu_final_out;
	cpu_final_out = new uint8_t[final_out_size];
	//cpu_shift(cpu_data_out, cpu_final_out, blk_offset, len_out, col_map, col_offset, chunk_offset);

	printf("setting final output size done\n");

	//free(col_offset);



	printf("cpu func done\n");

	*out = new uint8_t[final_out_size];
	printf("malloc out\n");



	cpu_shift(cpu_data_out, cpu_final_out, blk_offset, len_out, col_map, col_offset, chunk_offset, num_chunk);
	printf("shift done\n");

	std::ofstream col_map_file ("col_map.bin",std::ofstream::binary);
	col_map_file.write ((const char *)(col_map),  BLK_SIZE * num_chunk);
	col_map_file.close();

	std::ofstream col_len_file ("col_len.bin",std::ofstream::binary);
	col_len_file.write ((const char *)(len_out),  BLK_SIZE * num_chunk * 8);
	col_len_file.close();


	std::ofstream blk_off_file ("blk_offset.bin",std::ofstream::binary);
	blk_off_file.write ((const char *)(blk_offset), (num_chunk + 1)*8);
	blk_off_file.close();




//	uint64_t* out_64 = (uint64_t*) *out;
//	uint32_t* out_32 = (uint32_t*) (*out + 8);
	//out_64[0] = in_n_bytes;
	//out_32[0] = chunk_size;
	//out_32[1] = n_chunks;
	//out_32[2] = HIST_SIZE;
	//out_32[3] = MIN_MATCH_LENGTH;
	//out_32[4] = MAX_MATCH_LENGTH;
	//printf("in_n_bytes: %llu\tchunk_size: %llu\tn_chunks: %llu\tHIST_SIZE: %llu\tMIN_MATCH_LENGTH: %llu\tMAX_MATCH_LENGTH: %llu\tOFFSET_SIZE: %llu\tbitsNeeded(4096): %llu\n",
	  //     out_64[0], out_32[0], out_32[1], out_32[2], out_32[3], out_32[4], OFFSET_SIZE, bitsNeeded(HIST_SIZE));

	//cuda_err_chk(cudaMemcpy((*out), d_lens_out, len_bytes, cudaMemcpyDeviceToHost));

	
//	uint64_t out_data_bytes = ((uint64_t*)((*out) ))[n_chunks-1];
	//printf("out_data_bytes: %llu\n", out_data_bytes);

	//uint64_t out_data_bytes = chunk_offset[NUM_CHUNKS-1];
	//have to change
	uint64_t out_data_bytes = chunk_offset[num_chunk];
	printf("out_data_bytes: %llu\n", out_data_bytes);








//	cuda_err_chk(cudaMemcpy((*out) , temp, out_data_bytes, cudaMemcpyDeviceToHost));

//	memcpy((*out) +  head_bytes + len_bytes, cpu_data_out, out_data_bytes);
	// printf("cpu mem copy\n");
	memcpy((*out) , cpu_final_out, out_data_bytes);

	//printf("first byte: %p\n", ((*out) +  head_bytes + len_bytes)[0]);
	*out_n_bytes = head_bytes + len_bytes + out_data_bytes;

	 printf("first byte: %p\n", ((*out) )[0]);
        // *out_n_bytes = out_data_bytes;

	cuda_err_chk(cudaFree(d_out));
	cuda_err_chk(cudaFree(temp));
	printf("done");	
    }

    __host__ void decompress_gpu(const uint8_t* const in, uint8_t** out, const uint64_t in_n_bytes, uint64_t* out_n_bytes) {
	uint8_t* d_in;
	uint8_t* d_out;
	uint64_t* d_lens;

	const uint64_t* const in_64 = (const uint64_t*) in;

	*out_n_bytes = in_64[0];
	uint64_t out_bytes = *out_n_bytes;

	const uint32_t* const in_32 = (const uint32_t*) (in + sizeof(uint64_t));
	uint32_t chunk_size = in_32[0];
	uint32_t n_chunks = in_32[1];
	uint32_t hist_size = in_32[2];
	uint32_t min_match_len = in_32[3];
	uint32_t max_match_len = in_32[4];

	
	uint64_t len_bytes =  (n_chunks*sizeof(uint64_t));
	uint64_t head_bytes = HEAD_INTS*sizeof(uint32_t);
	//const uint64_t* const lens = (const uint64_t*) (in + head_bytes);
	const uint8_t* const in_ = in + head_bytes + len_bytes;
	uint64_t padded_in_n_bytes = in_n_bytes - (head_bytes + len_bytes);// + (chunk_size-(in_n_bytes % chunk_size));
	printf("first byte: %p\n", in_[0]);
	

	cuda_err_chk(cudaMalloc(&d_in, padded_in_n_bytes));
	cuda_err_chk(cudaMalloc(&d_lens, n_chunks * sizeof(uint64_t)));
	cuda_err_chk(cudaMalloc(&d_out, (*out_n_bytes)));
	printf("out_bytes: %p\td_out: %p\tchunk_size: %llu\tn_chunks: %llu\tHIST_SIZE: %llu\tMIN_MATCH_LENGTH: %llu\tMAX_MATCH_LENGTH: %llu\tOFFSET_SIZE: %llu\tbitsNeeded(4096): %llu\n",
	       out_bytes, d_out, chunk_size, n_chunks, hist_size, min_match_len, max_match_len, OFFSET_SIZE, bitsNeeded(HIST_SIZE));

	//return;
	
	cuda_err_chk(cudaMemcpy(d_in, in_, padded_in_n_bytes, cudaMemcpyHostToDevice));
	
	cuda_err_chk(cudaMemcpy(d_lens, in+head_bytes, len_bytes, cudaMemcpyHostToDevice));
	uint64_t grid_size = ceil<uint64_t>(n_chunks, BLK_SIZE);
        kernel_decompress<<<grid_size, BLK_SIZE>>>(d_in, d_out, padded_in_n_bytes, out_bytes, chunk_size, n_chunks, d_lens);
	cuda_err_chk(cudaDeviceSynchronize());
	

	printf("in bytes: %llu\n", in_n_bytes);

	*out = new uint8_t[out_bytes];
	uint32_t* out_32 = (uint32_t*) *out;
//	out_32[0] = HEAD_INTS ;
//	out_32[1] = chunk_size;
//	out_32[2] = n_chunks;
//	out_32[3] = HIST_SIZE;
//	out_32[4] = MIN_MATCH_LENGTH;
//	out_32[5] = MAX_MATCH_LENGTH;
	

	cuda_err_chk(cudaMemcpy((*out), d_out, out_bytes, cudaMemcpyDeviceToHost));
	printf("bytes: %llu\n",*out_n_bytes );


	cuda_err_chk(cudaFree(d_out));
	cuda_err_chk(cudaFree(d_in));
	cuda_err_chk(cudaFree(d_lens));
	
    }

}
