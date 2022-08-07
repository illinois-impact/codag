#ifndef _RLEV2_DECODER_TRANPOSE_H_
#define _RLEV2_DECODER_TRANPOSE_H_


#include "utils.h"
//#include "decoder_trans_two_warp.h"
//#include "decoder_trans_single_warp.h"
#include "decoder_new.h"

namespace rlev2 {
	template<int READ_UNIT, int queue_size, typename COMP_TYPE>
	__host__ void decompress_gpu(const uint8_t *in, const uint64_t in_n_bytes, const uint64_t n_chunks,
			blk_off_t *blk_off, col_len_t *col_len,
			INPUT_T *&out, uint64_t &out_n_bytes, uint64_t CHUNK_SIZE) {
		initialize_bit_maps();
		uint8_t *d_in;
		INPUT_T *d_out;
		blk_off_t *d_blk_off;
		col_len_t *d_col_len;

		auto exp_out_n_bytes = blk_off[n_chunks];
		const uint64_t exp_out_padded_bytes = ((exp_out_n_bytes - CHUNK_SIZE) / CHUNK_SIZE + 1) * CHUNK_SIZE;

		out_n_bytes = exp_out_n_bytes;

		cuda_err_chk(cudaMalloc(&d_in, in_n_bytes));
		cuda_err_chk(cudaMalloc(&d_out, exp_out_n_bytes + CHUNK_SIZE));
		cuda_err_chk(cudaMalloc(&d_blk_off, sizeof(blk_off_t) * n_chunks));
		cuda_err_chk(cudaMalloc(&d_col_len, sizeof(col_len_t) * n_chunks * BLK_SIZE));
			
		cuda_err_chk(cudaMemcpy(d_in, in, in_n_bytes, cudaMemcpyHostToDevice));
		cuda_err_chk(cudaMemcpy(d_blk_off, blk_off, sizeof(blk_off_t) * n_chunks, cudaMemcpyHostToDevice));
		cuda_err_chk(cudaMemcpy(d_col_len, col_len, sizeof(col_len_t) * n_chunks * BLK_SIZE, cudaMemcpyHostToDevice));

		std::chrono::high_resolution_clock::time_point kernel_start = std::chrono::high_resolution_clock::now();
		// decompress_func_write_sync<<<n_chunks, dim3(BLK_SIZE, 2, 1)>>>(d_in, n_chunks, d_blk_off, d_col_len, d_out);
		//decompress_func_read_sync<READ_UNIT><<<n_chunks, dim3(BLK_SIZE, 2, 1)>>>(d_in, n_chunks, d_blk_off, d_col_len, d_out, CHUNK_SIZE);
		inflate<uint32_t, COMP_TYPE, COMP_TYPE, uint8_t, queue_size><<<n_chunks, dim3(BLK_SIZE, 2, 1)>>> (d_in, (uint8_t*) d_out, (uint64_t*)d_col_len, (uint64_t*)d_blk_off	,  4, READ_UNIT, CHUNK_SIZE);
			
		// decompress_single_warp<READ_UNIT, 32>(d_in, n_chunks, d_blk_off, d_col_len, d_out);
		// decompress_kernel_single_warp<READ_UNIT><<<n_chunks, BLK_SIZE>>>(d_in, n_chunks, d_blk_off, d_col_len, d_out);
		cuda_err_chk(cudaDeviceSynchronize());
		std::chrono::high_resolution_clock::time_point kernel_end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> total = std::chrono::duration_cast<std::chrono::duration<double>>(kernel_end - kernel_start);		

		std::cout << total.count() << " secs\n";
	
		out = new INPUT_T[exp_out_n_bytes / sizeof(INPUT_T)];
		cuda_err_chk(cudaMemcpy(out, d_out, exp_out_n_bytes, cudaMemcpyDeviceToHost));
		
		cuda_err_chk(cudaFree(d_in));
		cuda_err_chk(cudaFree(d_out));
		cuda_err_chk(cudaFree(d_blk_off));
		cuda_err_chk(cudaFree(d_col_len));
	}

	template<int READ_UNIT, int queue_size, typename COMP_TYPE, int NUM_WARP = 2, int NUM_CHUNKS = 1>
	__host__ void decompress_gpu_orig(const uint8_t *in, const uint64_t in_n_bytes, const uint64_t n_chunks,
			blk_off_t *blk_off, col_len_t *col_len,
			INPUT_T *&out, uint64_t &out_n_bytes, uint64_t CHUNK_SIZE) {
		initialize_bit_maps();
		uint8_t *d_in;
		INPUT_T *d_out;
		blk_off_t *d_blk_off;
		col_len_t *d_col_len;

		auto exp_out_n_bytes = blk_off[n_chunks];
		const uint64_t exp_out_padded_bytes = ((exp_out_n_bytes - CHUNK_SIZE) / CHUNK_SIZE + 1) * CHUNK_SIZE;

		out_n_bytes = exp_out_n_bytes;

		cuda_err_chk(cudaMalloc(&d_in, in_n_bytes));
		cuda_err_chk(cudaMalloc(&d_out, exp_out_n_bytes + CHUNK_SIZE));
		cuda_err_chk(cudaMalloc(&d_blk_off, sizeof(blk_off_t) * n_chunks));
		cuda_err_chk(cudaMalloc(&d_col_len, sizeof(col_len_t) * n_chunks * BLK_SIZE));
			
		cuda_err_chk(cudaMemcpy(d_in, in, in_n_bytes, cudaMemcpyHostToDevice));
		cuda_err_chk(cudaMemcpy(d_blk_off, blk_off, sizeof(blk_off_t) * n_chunks, cudaMemcpyHostToDevice));
		cuda_err_chk(cudaMemcpy(d_col_len, col_len, sizeof(col_len_t) * n_chunks * BLK_SIZE, cudaMemcpyHostToDevice));

		cudaMemset(d_out, 0, exp_out_n_bytes + CHUNK_SIZE);

	//		printf("n_chunks: ");
		uint64_t num_blk = n_chunks / NUM_CHUNKS;

		std::chrono::high_resolution_clock::time_point kernel_start = std::chrono::high_resolution_clock::now();

		if(NUM_WARP == 2){
			if(NUM_CHUNKS == 1)
				inflate_orig_dw<uint32_t, COMP_TYPE, COMP_TYPE, uint8_t, NUM_CHUNKS, queue_size><<<num_blk, dim3(BLK_SIZE, 1 + NUM_CHUNKS, 1)>>> (d_in, (uint8_t*) d_out, (uint64_t*)d_col_len, (uint64_t*)d_blk_off	,  4, READ_UNIT, CHUNK_SIZE, n_chunks);
			else if(NUM_CHUNKS == 2)
				inflate_orig_dw<uint32_t, COMP_TYPE, COMP_TYPE, uint8_t, NUM_CHUNKS, queue_size, 96, 21><<<num_blk, dim3(BLK_SIZE, 1 + NUM_CHUNKS, 1)>>> (d_in, (uint8_t*) d_out, (uint64_t*)d_col_len, (uint64_t*)d_blk_off	,  4, READ_UNIT, CHUNK_SIZE, n_chunks);
			else if(NUM_CHUNKS == 4)
				inflate_orig_dw<uint32_t, COMP_TYPE, COMP_TYPE, uint8_t, NUM_CHUNKS, queue_size, 160, 12><<<num_blk, dim3(BLK_SIZE, 1 + NUM_CHUNKS, 1)>>> (d_in, (uint8_t*) d_out, (uint64_t*)d_col_len, (uint64_t*)d_blk_off	,  4, READ_UNIT, CHUNK_SIZE, n_chunks);
			else if(NUM_CHUNKS == 8)
				inflate_orig_dw<uint32_t, COMP_TYPE, COMP_TYPE, uint8_t, NUM_CHUNKS, queue_size, 288, 7><<<num_blk, dim3(BLK_SIZE, 1 + NUM_CHUNKS, 1)>>> (d_in, (uint8_t*) d_out, (uint64_t*)d_col_len, (uint64_t*)d_blk_off	,  4, READ_UNIT, CHUNK_SIZE, n_chunks);
				else if(NUM_CHUNKS == 16)
				inflate_orig_dw<uint32_t, COMP_TYPE, COMP_TYPE, uint8_t, NUM_CHUNKS, queue_size, 544, 3><<<num_blk, dim3(BLK_SIZE, 1 + NUM_CHUNKS, 1)>>> (d_in, (uint8_t*) d_out, (uint64_t*)d_col_len, (uint64_t*)d_blk_off	,  4, READ_UNIT, CHUNK_SIZE, n_chunks);
		}
	 	else if(NUM_WARP == 1){
			if(NUM_CHUNKS == 1){
				inflate_orig_rdw<uint32_t, COMP_TYPE, COMP_TYPE, uint8_t, NUM_CHUNKS, queue_size, 32, 32><<<num_blk, dim3(BLK_SIZE, 1, 1)>>> (d_in, (uint8_t*) d_out, (uint64_t*)d_col_len, (uint64_t*)d_blk_off	,  4, READ_UNIT, CHUNK_SIZE, n_chunks);
			}
			else if(NUM_CHUNKS == 2){
				inflate_orig_rdw<uint32_t, COMP_TYPE, COMP_TYPE, uint8_t, NUM_CHUNKS, queue_size, 64, 16><<<num_blk, dim3(BLK_SIZE, NUM_CHUNKS, 1)>>> (d_in, (uint8_t*) d_out, (uint64_t*)d_col_len, (uint64_t*)d_blk_off	,  4, READ_UNIT, CHUNK_SIZE, n_chunks);
			}
			else if(NUM_CHUNKS == 4){
				inflate_orig_rdw<uint32_t, COMP_TYPE, COMP_TYPE, uint8_t, NUM_CHUNKS, queue_size, 128, 8><<<num_blk, dim3(BLK_SIZE, NUM_CHUNKS, 1)>>> (d_in, (uint8_t*) d_out, (uint64_t*)d_col_len, (uint64_t*)d_blk_off	,  4, READ_UNIT, CHUNK_SIZE, n_chunks);
			}
			else if(NUM_CHUNKS == 8){
				inflate_orig_rdw<uint32_t, COMP_TYPE, COMP_TYPE, uint8_t, NUM_CHUNKS, queue_size, 256, 4><<<num_blk, dim3(BLK_SIZE, NUM_CHUNKS, 1)>>> (d_in, (uint8_t*) d_out, (uint64_t*)d_col_len, (uint64_t*)d_blk_off	,  4, READ_UNIT, CHUNK_SIZE, n_chunks);
			}
			else if(NUM_CHUNKS == 16){
				inflate_orig_rdw<uint32_t, COMP_TYPE, COMP_TYPE, uint8_t, NUM_CHUNKS, queue_size, 512, 2><<<num_blk, dim3(BLK_SIZE, NUM_CHUNKS, 1)>>> (d_in, (uint8_t*) d_out, (uint64_t*)d_col_len, (uint64_t*)d_blk_off	,  4, READ_UNIT, CHUNK_SIZE, n_chunks);
			}
			
		}


		cuda_err_chk(cudaDeviceSynchronize());
		std::chrono::high_resolution_clock::time_point kernel_end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> total = std::chrono::duration_cast<std::chrono::duration<double>>(kernel_end - kernel_start);		

		std::cout << total.count() << " secs\n";
	
		out = new INPUT_T[exp_out_n_bytes / sizeof(INPUT_T)];
		cuda_err_chk(cudaMemcpy(out, d_out, exp_out_n_bytes, cudaMemcpyDeviceToHost));
		
		cuda_err_chk(cudaFree(d_in));
		cuda_err_chk(cudaFree(d_out));
		cuda_err_chk(cudaFree(d_blk_off));
		cuda_err_chk(cudaFree(d_col_len));
	}

}
#endif
