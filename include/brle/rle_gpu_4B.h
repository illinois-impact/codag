
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
constexpr   uint64_t CHUNK_SIZE_() { return (4*1024*2); }

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

#define char_len							4

namespace brle_trans {

	__global__ void __launch_bounds__( 128, 16 )
  decompress_func(const uint8_t* const in, uint8_t* out,
    							    const uint64_t n_chunks,
    							    uint64_t* col_len,  uint8_t* col_map, uint64_t* blk_offset){

		__shared__ uint64_t shared_col_len[BLK_SIZE*4];

		int tid = threadIdx.x % 32;
	    int chunk_idx = blockIdx.x * 4 + (threadIdx.x / 32);


//	    int tid = threadIdx.x;
//	    int chunk_idx = blockIdx.x;
	    shared_col_len[threadIdx.x] = col_len[BLK_SIZE * chunk_idx + tid];
    	
    	__syncthreads();


    	int block_flag = (threadIdx.x / 32);

	   	uint64_t used_bytes = 0;
	   	uint64_t mychunk_size = col_len[BLK_SIZE * chunk_idx + tid];

	    uint64_t in_start_idx = blk_offset[chunk_idx];
	    
	    uint64_t col_counter = BLK_SIZE;
	    
	    uint8_t head_byte = 1;
	    uint8_t block = 0;


	    uint64_t out_off = 0;

	    uint64_t out_start_idx = chunk_idx * CHUNK_SIZE;

	    bool compress_flag = false;
	    uint8_t compress_counter;

	    uint64_t col_idx = col_map[BLK_SIZE * chunk_idx + tid];


		uint8_t v = 0;

		uint32_t data_buffer;
		uint32_t* in_4B = (uint32_t *)(&(in[in_start_idx]));
		uint32_t in_4B_off = 0;
		uint32_t in_counter = 4;
		uint8_t c_flag = 0;

		int data_counter = 0;

		uint32_t temp_buffer = 0;

		uint32_t* out_4B = (uint32_t*)(&(out[out_start_idx + col_idx*4]));


		while(used_bytes  < mychunk_size){

			c_flag = (head_byte & 1);

			
			if(in_counter == 4){
				data_buffer = in_4B[in_4B_off + tid];
				data_counter++;
				while((data_counter * 4 > (((shared_col_len[block_flag * 32 + col_counter - 1] + 3)/4)*4)) && (col_counter > 1)){
							col_counter--;
				}

				in_4B_off += col_counter;
				in_counter = 0;		
				
			}
		

			if(block == 0){
				v = ((uint8_t*)(&data_buffer)) [in_counter];
				head_byte = v;
				block++;

				in_counter++;
				used_bytes++;

				if(in_counter == 4){
					data_buffer = in_4B[in_4B_off + tid];
					data_counter++;

					while((data_counter * 4 > (((shared_col_len[block_flag * 32 + col_counter - 1] + 3)/4)*4)) && (col_counter > 1)){
								col_counter--;
					}

					in_4B_off += col_counter;
					in_counter = 0;
					
				}
			}

			else if(c_flag == 0 && compress_flag == false){
				v = ((uint8_t*)(&data_buffer)) [in_counter];
				compress_flag = true;
				compress_counter = v;

				in_counter++;
				used_bytes++;

				if(in_counter == 4){
					data_buffer = in_4B[in_4B_off + tid];
					data_counter++;

					while((data_counter * 4 > (((shared_col_len[block_flag * 32 + col_counter - 1] + 3)/4)*4)) && (col_counter > 1)){
								col_counter--;
					}

					in_4B_off += col_counter;						
					in_counter = 0;
				
				}	

		

			}


			else {

					int temp_counter = 0;
					for(int i = in_counter; i<4; i++){
						((uint8_t*)(&temp_buffer))[temp_counter] =  ((uint8_t*)(&data_buffer)) [i];
						temp_counter++;
					}
					data_buffer = in_4B[in_4B_off + tid];
					data_counter++;

					while((data_counter * 4 > (((shared_col_len[block_flag * 32 + col_counter - 1] + 3)/4)*4)) && (col_counter > 1)){
									col_counter--;
					}

					in_4B_off += col_counter;	
					
					for(int i = 0; i < in_counter; i++){
						 ((uint8_t*)(&temp_buffer))[temp_counter] =  ((uint8_t*)(&data_buffer)) [i];
						temp_counter++;
					}
				

					used_bytes += 4;

		    		head_byte = head_byte >> 1;
					block++;
					if(block == 9){
						block = 0;
					}



				if(compress_flag == true){


					for(uint8_t k = 0; k < compress_counter; k++){	
						out_4B[out_off] = temp_buffer;
						out_off+=32;
					}

					compress_flag = false;

				}
				else{
					out_4B[out_off] = temp_buffer;
					out_off+=32;

				}
			}
		
		}


	}

	__global__ void ExampleKernel(uint64_t* col_len, uint8_t* col_map, uint64_t* out)
	{
	    // Specialize BlockRadixSort for a 1D block of 128 threads owning 4 integer keys and values each
	    typedef cub::BlockRadixSort<uint64_t, 32, 1, int> BlockRadixSort;
	    __shared__ typename BlockRadixSort::TempStorage temp_storage;

	    int tid = threadIdx.x;
	    int bid = blockIdx.x;

	    uint64_t thread_keys[1];
	    int thread_values[1];
	    thread_keys[0] = col_len[bid*BLK_SIZE + tid];
	    thread_values[0] = tid;

	    BlockRadixSort(temp_storage).Sort(thread_keys, thread_values);
	    
		//col_len[bid*BLK_SIZE + tid] = thread_keys[0];
		col_map[bid*BLK_SIZE + BLK_SIZE - 1 - tid] = thread_values[0];
		out[bid*BLK_SIZE + BLK_SIZE - 1 - tid] = thread_keys[0];
	}

		//change it to actual parallel scan
	__global__ void parallel_scan(uint64_t* blk_offset, uint64_t n){
		for(int i = 1; i <= n;i++){
			blk_offset[i] += blk_offset[i-1];
		}
	}


	__device__ bool check_f (uint8_t* a, uint8_t*b){

	 	for(int i = 0; i < char_len; i++){
	 		if(a[i] != b[i]){
	 			return false;
	 		}
	 	}
	 	return true;


	}

	__global__ void compress_init_func(const uint8_t* const in, const uint64_t in_n_bytes,  
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


    	while(used_bytes < mychunk_size){

    		for(int i = 0; i < char_len; i++){
				v_buffer[i] = in[in_start_idx + used_bytes + in_off + tid * 4];
				
				in_flag++;
    			if(in_flag == 4){
    				in_off += 31 * 4;
    				in_flag = 0;
    			}
    			used_bytes++;
			}

    		if(counter == 0){
    			for(int i = 0; i < char_len; i++){
    				prev_buffer[i] = v_buffer[i];
    			}
    		}

    		if(check_f(prev_buffer,v_buffer)){
    			counter++;
    		}

    		else{
    			if(counter >= 3){
    				out_len += 5;
    				counter = 1;
    				
					for(int i = 0; i < char_len; i++){
    					prev_buffer[i] = v_buffer[i];
    				}

    				pcounter++;

    				if ((++blocks) == 8) {
    						out_len ++;
						    blocks = 0;
					}
    			}

    			else{

    				for(uint8_t j = 0; j < counter; j++){
    					out_len+=4;
    					reg_counter++;

		   				if ((++blocks) == 8) {
						    out_len++;
						    blocks = 0;
						}
    				}
    				counter = 1;
    				for(int i = 0; i < char_len; i++){
    					prev_buffer[i] = v_buffer[i];
    				}

    			}
    		}

    	}  


		if(counter >= 3){
			pcounter ++;
			out_len+=5;
		}  
		else{
			for(uint8_t j = 0; j < counter; j++){
    					out_len+=4;
    					reg_counter++;

		   				if ((++blocks) == 8) {
						    out_len++;
						    blocks = 0;
						}
    				}

		}
	
    	
    
		col_len[BLK_SIZE*chunk_idx + tid] = out_len; 


		uint64_t out_len_4B = ((out_len + 3)/4)*4;
	    atomicAdd((unsigned long long int *)&block_len, (unsigned long long int )out_len_4B);
	 	 __syncthreads();
	 	if(threadIdx.x == 0){
	  		//128B alignment
	  		block_len = ((block_len + 127)/128)*128;
	  		blk_offset[chunk_idx+1] = (uint64_t)block_len;
	    }
	}





	 __device__ void writeupdate (uint64_t* byte_counter_p, uint64_t* byte_offset_p, uint64_t* col_len, uint64_t* col_counter_p, int chunk_idx) {

	 	byte_counter_p[0]++;
	 	byte_offset_p[0]++;

		if((byte_counter_p[0] %4 ==0) && (col_counter_p[0] != 0)){
				while((byte_counter_p[0] > ((col_len[chunk_idx * BLK_SIZE + col_counter_p[0]] + 3)/4) * 4) &&(col_counter_p[0] > 0)){
					col_counter_p[0]--;
				}
				byte_offset_p[0] += col_counter_p[0] * 4;
	   		}

	 	}


	 


	__global__ void compress_func (const uint8_t* const in, uint8_t* out, const uint64_t in_n_bytes, uint64_t* out_n_bytes, const uint64_t in_chunk_size, 
    							    const uint64_t n_chunks,
    							   uint64_t* col_len,  uint8_t* col_map, uint64_t* blk_offset){

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


    	for(int i = 0; i < 32;i++){
			if (tid ==  col_map[BLK_SIZE * chunk_idx + i]){
				col_idx = i;
			}
		}



		while(used_bytes < mychunk_size){

			for(int i = 0; i < char_len; i++){
				v_buffer[i] = in[in_start_idx + used_bytes + in_off + tid * 4];
				
				in_flag++;
    			if(in_flag == 4){
    				in_off += 31 * 4;
    				in_flag = 0;
    			}
    			used_bytes++;
			}
  
    		
    		if(counter == 0){
    			for(int i = 0; i < char_len; i++){
    				prev_buffer[i] = v_buffer[i];
    			}
    		}

    		if(check_f(prev_buffer, v_buffer)){
    			counter++;
    		}

    		else{

    	
				if(counter >= 3){

					out[out_start_idx + byte_offset + col_idx * 4] = counter;

					writeupdate (&byte_counter, &byte_offset, col_len, 
								 &col_counter, chunk_idx);

					for(int i = 0; i < char_len; i++){

						out[out_start_idx + byte_offset + col_idx * 4] = prev_buffer[i];
				
						writeupdate (&byte_counter, &byte_offset, col_len, 
								 &col_counter, chunk_idx);
					}

    				

    				counter = 1;

    				for(int i = 0; i < char_len; i++){
    					prev_buffer[i] = v_buffer[i];
    				}


    				if ((++blocks) == 8) {
    						
						    out[out_start_idx + cur_header_byte_pos + col_idx * 4] = header_byte;
						    header_byte = 0;
						 

						    cur_header_byte_pos = byte_offset;

						   
							writeupdate (&byte_counter, &byte_offset, col_len, 
								         &col_counter, chunk_idx);
						    
						    blocks = 0;

					}

					if (blocks != 0) {
							out[out_start_idx + cur_header_byte_pos + col_idx * 4] = header_byte;
					 }
    			}

    			else{
    				for(int j = 0; j < counter; j++){
    					header_byte = (header_byte | (1 << (blocks)));


    					for(int i = 0; i < char_len; i++){
							out[out_start_idx + byte_offset + col_idx * 4] = prev_buffer[i];
    					
							writeupdate (&byte_counter, &byte_offset, col_len, 
									 &col_counter, chunk_idx);

    					}
    	
		   				if ((++blocks) == 8) {

						    out[out_start_idx + cur_header_byte_pos + col_idx * 4] = header_byte;
						    header_byte = 0;
						 

						    cur_header_byte_pos = byte_offset;
							writeupdate (&byte_counter, &byte_offset, col_len, 
									 &col_counter, chunk_idx);
						    
						    blocks = 0;
						}
						if (blocks != 0) {
							out[out_start_idx + cur_header_byte_pos + col_idx * 4] = header_byte;
					     }
					 }

					 
					counter = 1;
    				
    				for(int i = 0; i < char_len; i++){
    					prev_buffer[i] = v_buffer[i];
    				}

	    		}

    		}
	}


	//fix it
	if(counter >= 3){
		out[out_start_idx + byte_offset + col_idx * 4] = counter;
		writeupdate (&byte_counter, &byte_offset, col_len, 
									 &col_counter, chunk_idx);

		for(int i = 0; i < char_len; i++){
			out[out_start_idx + byte_offset + col_idx * 4] = prev_buffer[i];
			writeupdate (&byte_counter, &byte_offset, col_len, 
									 &col_counter, chunk_idx);
		}

	}  
	else{

		for(uint8_t j = 0; j < counter; j++){
			header_byte = (header_byte | (1 << (blocks)));


			for(int i = 0; i < char_len; i++){

				out[out_start_idx + byte_offset + col_idx * 4] = prev_buffer[i];
		

				writeupdate (&byte_counter, &byte_offset, col_len, 
									 &col_counter, chunk_idx);
			}

			if ((++blocks) == 8) {

			    out[out_start_idx + cur_header_byte_pos + col_idx * 4] = header_byte;
			    header_byte = 0;
			 

			    cur_header_byte_pos = byte_offset;

			    writeupdate (&byte_counter, &byte_offset, col_len, 
									 &col_counter, chunk_idx);
			    blocks = 0;
			}
			if (blocks != 0) {
				out[out_start_idx + cur_header_byte_pos + col_idx * 4] = header_byte;
		     }

		 }

	}



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


	printf("in bytes: %llu\n", in_n_bytes);


	uint64_t num_chunk = in_n_bytes / CHUNK_SIZE;
	//printf("cpu num chunk: %llu\n", num_chunk);



	//cpu
	uint8_t* cpu_data_out = (uint8_t*) malloc(exp_data_out_bytes);
	uint64_t* col_len = (uint64_t*) malloc(sizeof(uint64_t) * BLK_SIZE * num_chunk);
	uint8_t* col_map = (uint8_t*) malloc(BLK_SIZE * num_chunk);
	uint64_t* blk_offset = (uint64_t*) malloc(8*(num_chunk + 1));
	uint64_t* chunk_offset = (uint64_t*) malloc(8*(num_chunk + 1));
	uint64_t* col_offset = (uint64_t*) malloc(8*(BLK_SIZE * num_chunk + 1));


	uint64_t* d_blk_offset;
	uint64_t* d_col_len;
	uint8_t*  d_col_map;
	uint64_t* d_col_len_sorted;


	cuda_err_chk(cudaMalloc(&d_in, padded_in_n_bytes));
	cuda_err_chk(cudaMalloc(&d_col_len, sizeof(uint64_t) * BLK_SIZE * num_chunk));
	cuda_err_chk(cudaMalloc(&d_col_len_sorted, sizeof(uint64_t) * BLK_SIZE * num_chunk));

	cuda_err_chk(cudaMalloc(&d_col_map, BLK_SIZE * num_chunk));
	cuda_err_chk(cudaMalloc(&d_blk_offset, 8*(num_chunk + 1)));

	cuda_err_chk(cudaMemcpy(d_in, in, in_n_bytes, cudaMemcpyHostToDevice));

	compress_init_func<<<n_chunks, BLK_SIZE>>>(d_in, in_n_bytes, 
											  chunk_size,exp_out_chunk_size, n_chunks,
											  d_col_len,  d_blk_offset);
	cuda_err_chk(cudaDeviceSynchronize()); 

	parallel_scan<<<1,1>>>(d_blk_offset, n_chunks);
	cuda_err_chk(cudaDeviceSynchronize());
	ExampleKernel<<<n_chunks, BLK_SIZE>>>(d_col_len, d_col_map, d_col_len_sorted);
	cuda_err_chk(cudaDeviceSynchronize());


	cuda_err_chk(cudaMemcpy(col_len, d_col_len_sorted, sizeof(uint64_t) * BLK_SIZE * num_chunk, cudaMemcpyDeviceToHost));
	cuda_err_chk(cudaMemcpy(blk_offset, d_blk_offset,  8*(num_chunk + 1), cudaMemcpyDeviceToHost));
	cuda_err_chk(cudaMemcpy(col_map, d_col_map, BLK_SIZE * num_chunk, cudaMemcpyDeviceToHost));


	uint64_t final_out_size = blk_offset[num_chunk];
	*out = new uint8_t[final_out_size];

	cuda_err_chk(cudaMalloc(&d_out, final_out_size));


	compress_func<<<n_chunks, BLK_SIZE>>>(d_in, d_out, in_n_bytes, out_n_bytes, chunk_size, n_chunks,
										  d_col_len_sorted, d_col_map, d_blk_offset);
	cuda_err_chk(cudaDeviceSynchronize());
	cuda_err_chk(cudaMemcpy((*out), d_out, final_out_size, cudaMemcpyDeviceToHost));


	std::ofstream col_len_file ("./input_data/col_len.bin",std::ofstream::binary);
	col_len_file.write ((const  char *)(col_len),  BLK_SIZE * num_chunk * 8);
	col_len_file.close();


	std::ofstream blk_off_file ("./input_data/blk_offset.bin",std::ofstream::binary);
	blk_off_file.write ((const char *)(blk_offset), (num_chunk + 1)*8);
	blk_off_file.close();


	std::ofstream col_map_file ("./input_data/col_map.bin",std::ofstream::binary);
	col_map_file.write ((const char *)(col_map),  BLK_SIZE * num_chunk);
	col_map_file.close();






	*out_n_bytes = final_out_size;
	cuda_err_chk(cudaFree(d_out));
	cuda_err_chk(cudaFree(d_col_len));
	cuda_err_chk(cudaFree(d_col_map));
	cuda_err_chk(cudaFree(d_col_len_sorted));
	cuda_err_chk(cudaFree(d_in));
	cuda_err_chk(cudaFree(d_blk_offset));





    }

    __host__ void decompress_gpu(const uint8_t* const in, uint8_t** out, const uint64_t in_n_bytes, uint64_t* out_n_bytes) {



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

  	if((fd_col_len = open(filename_col_len, O_RDONLY)) == -1){
        printf("Fatal Error: Col Len read error\n");
        return;
    }
    
    if((fd_col_map = open(filename_col_map, O_RDONLY)) == -1){
        printf("Fatal Error: Col map read error\n");
        return;
    }

 	if((fd_blk_off = open(filename_blk_off, O_RDONLY)) == -1){
        printf("Fatal Error: Block off read error\n");
        return;
    }

 	fstat(fd_col_len, &sbcol_len);
    fstat(fd_col_map, &sbcol_map);
    fstat(fd_blk_off, &sbblk_off);

    void* map_base_col_len;
    void* map_base_col_map;
    void* map_base_blk_off;

    map_base_col_len = mmap(NULL, sbcol_len.st_size, PROT_READ, MAP_SHARED, fd_col_len, 0);
    map_base_col_map = mmap(NULL, sbcol_map.st_size, PROT_READ, MAP_SHARED, fd_col_map, 0);
    map_base_blk_off = mmap(NULL, sbblk_off.st_size, PROT_READ, MAP_SHARED, fd_blk_off, 0);


    uint64_t num_blk = ((uint64_t) sbblk_off.st_size / sizeof(uint64_t)) - 1;
    //uint64_t blk_size = ((uint8_t) sbcol_map.st_size / num_blk);
    uint64_t blk_size = BLK_SIZE;



    //start
    std::chrono::high_resolution_clock::time_point kernel_start = std::chrono::high_resolution_clock::now();


    uint8_t* d_in;
    uint8_t* d_out;

    uint64_t* d_col_len;
    uint64_t* d_blk_offset;
    uint8_t* d_col_map;
	
	const uint8_t* const in_ = in ;

    //change it later
    uint64_t in_bytes = ((uint64_t*)map_base_blk_off)[num_blk];
    uint64_t out_bytes = CHUNK_SIZE * num_blk;
    *out_n_bytes = out_bytes;

    printf("out_bytes: %llu\n", out_bytes);


	cuda_err_chk(cudaMalloc(&d_in, in_bytes));
	cuda_err_chk(cudaMalloc(&d_out, (*out_n_bytes)));

	cuda_err_chk(cudaMalloc(&d_col_len, sbcol_len.st_size));
	cuda_err_chk(cudaMalloc(&d_col_map, sbcol_map.st_size));
	cuda_err_chk(cudaMalloc(&d_blk_offset, sbblk_off.st_size));


	cuda_err_chk(cudaMemcpy(d_in, in_, in_bytes, cudaMemcpyHostToDevice));

    cuda_err_chk(cudaMemcpy(d_col_len, map_base_col_len, sbcol_len.st_size, cudaMemcpyHostToDevice));
    cuda_err_chk(cudaMemcpy(d_col_map, map_base_col_map, sbcol_map.st_size, cudaMemcpyHostToDevice));
    cuda_err_chk(cudaMemcpy(d_blk_offset, map_base_blk_off, sbblk_off.st_size, cudaMemcpyHostToDevice));

    printf("cuda malloc finished\n");
    printf("num_blk: %llu, blk_size: %llu\n", num_blk, blk_size);


    decompress_func<<<(num_blk/4), blk_size * 4>>> (d_in, d_out, num_blk, d_col_len, d_col_map, d_blk_offset);
     // decompress_func<<<(num_blk), blk_size >>> (d_in, d_out, num_blk, d_col_len, d_col_map, d_blk_offset);
      cudaDeviceSynchronize();
       printf("decomp function done\n");

       cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }


	*out = new uint8_t[out_bytes];
	cuda_err_chk(cudaMemcpy((*out), d_out, out_bytes, cudaMemcpyDeviceToHost));

	std::chrono::high_resolution_clock::time_point kernel_end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> kt = std::chrono::duration_cast<std::chrono::duration<double>>(kernel_end - kernel_start);
	std::cout << "Decompression time: " << kt.count() << " secs\n";


    if(munmap(map_base_col_len, sbcol_len.st_size) == -1){
    	printf("Mem unmap error");
    }
    if(munmap(map_base_col_map, sbcol_map.st_size) == -1){
    	printf("Mem unmap error");
    }

    if(munmap(map_base_blk_off, sbblk_off.st_size) == -1){    	
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
}

