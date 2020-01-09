#include <common.h>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

constexpr   uint16_t THRDS_SM_() { return (2048); }
constexpr   uint16_t BLK_SIZE_() { return (1024); }
constexpr   uint16_t BLKS_SM_()  { return (THRDS_SM_()/BLK_SIZE_()); }
constexpr   uint64_t GRID_SIZE_() { return (1024); }
constexpr   uint64_t NUM_CHUNKS_() { return (GRID_SIZE_()*BLK_SIZE_()); }
constexpr   uint64_t CHUNK_SIZE_() { return (256*1024); }
constexpr   uint64_t HEADER_SIZE_() { return (1); }
constexpr   uint32_t OVERHEAD_PER_CHUNK_(uint32_t d) { return (ceil<uint32_t>(d,(HEADER_SIZE_()*8))+1); } 
constexpr   uint32_t HIST_SIZE_() { return 4096; }
constexpr   uint32_t LOOKAHEAD_SIZE_() { return 4096; }
constexpr   uint32_t OFFSET_SIZE_() { return (bitsNeeded((uint32_t)HIST_SIZE_())); }
constexpr   uint32_t LENGTH_SIZE_() { return (4); }
constexpr   uint32_t MIN_MATCH_LENGTH_() { return (ceil<uint32_t>((OFFSET_SIZE_()+LENGTH_SIZE_()),8)+1); }
constexpr   uint32_t MAX_MATCH_LENGTH_() { return (pow<uint32_t, uint32_t>(2,LENGTH_SIZE_()) + MIN_MATCH_LENGTH_()); }
constexpr   uint8_t DEFAULT_CHAR_() { return ' '; }
constexpr   uint32_t HEAD_INTS_() { return 6; }
constexpr   uint32_t READ_UNITS_() { return 1; }

constexpr   uint32_t LOOKAHEAD_UNITS_() { return LOOKAHEAD_SIZE_()/READ_UNITS_(); }

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
#define MIN_MATCH_LENGTH		  MIN_MATCH_LENGTH_()		  
#define MAX_MATCH_LENGTH		  MAX_MATCH_LENGTH_()		  
#define DEFAULT_CHAR			  DEFAULT_CHAR_()			  
#define HEAD_INTS                         HEAD_INTS_()
#define READ_UNITS                        READ_UNITS_()
#define LOOKAHEAD_UNITS                   LOOKAHEAD_UNITS_()


namespace lzss {
    __host__ __device__ void find_match(const uint8_t* const  hist, const uint32_t hist_head, const uint32_t hist_count, const uint8_t* const lookahead, const uint32_t lookahead_head, const uint32_t lookahead_count, uint32_t* offset, uint32_t* length) {
	uint32_t hist_offset = 0;
	uint32_t f_length = 0;
	uint32_t max_len = 0;
	uint32_t max_offset = 0;
	while (hist_offset < hist_count) {
	    if (hist[(hist_head+hist_offset) % HIST_SIZE] == lookahead[(lookahead_head)]) {
		f_length = 1;

		while (((hist_offset + f_length) < hist_count) && ((f_length) < lookahead_count) &&
		       (hist[(hist_head + hist_offset + f_length) % HIST_SIZE] ==
			lookahead[(lookahead_head + f_length) % LOOKAHEAD_SIZE])) {
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
	*offset = max_offset;

    }

    __host__ __device__ void compress_func_old(const uint8_t* const in, uint8_t* out, const uint64_t in_n_bytes, const uint64_t out_n_bytes, const uint64_t in_chunk_size, const uint64_t out_chunk_size, const uint64_t n_chunks, uint64_t* lens, const uint64_t tid) {
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
	    uint32_t consumed_bytes = 0;
	    uint32_t out_bytes = 1;
	    uint32_t cur_header_byte_pos = 0;
	    uint8_t header_byte = 0;
	    uint8_t blocks = 0;
	    uint32_t c = 0;

	    uint32_t used_bytes = 0;

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
			hist_head = (hist_head + (HIST_SIZE-hist_count)) % HIST_SIZE;
			hist_count = HIST_SIZE;
		    }
		    header_byte = (header_byte | 1);
		    used_bytes++;
		    
		}
		//match
		else {
		    uint64_t v = (offset << LENGTH_SIZE) | (length - MIN_MATCH_LENGTH);

		    for (size_t i = 0; i < (MIN_MATCH_LENGTH-1); i++) {
			out[out_start_idx + (out_bytes++)] = v & 0x00FF;
			v >>= 8;
		    }
		    uint32_t hist_start = hist_head+hist_count;
		    for (size_t i = 0; i < length; i++) {
			hist[(hist_start+i) % HIST_SIZE] = lookahead[(lookahead_head+i) % HIST_SIZE];
		    }
		    
		    lookahead_head = (lookahead_head + length) % LOOKAHEAD_SIZE;
		    lookahead_count -= length;
		    hist_count += length;
		    if (hist_count > HIST_SIZE) {
			hist_head = (hist_head + (HIST_SIZE-hist_count)) % HIST_SIZE;
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
		else
		    header_byte <<= 1;
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
    
    __host__ __device__ void compress_func(const uint8_t* const in, uint8_t* out, const uint64_t in_n_bytes, const uint64_t out_n_bytes, const uint64_t in_chunk_size, const uint64_t out_chunk_size, const uint64_t n_chunks, uint64_t* lens, const uint64_t tid) {
	if (tid < n_chunks) {
	    uint64_t rem = in_n_bytes % in_chunk_size;
	    uint64_t my_chunk_size = ((tid == (n_chunks - 1)) && (rem)) ? rem : in_chunk_size;
	    uint64_t in_start_idx = tid * (in_chunk_size/READ_UNITS);
	    uint64_t out_start_idx = tid * out_chunk_size;

	    uint8_t hist[HIST_SIZE] = {DEFAULT_CHAR};
	    
	    
	    uint8_t lookahead[LOOKAHEAD_SIZE];
	    uint8_t* lookahead_4 = (uint8_t*) lookahead;
	    uint32_t hist_head  = 0;
	    uint32_t hist_count = 0;
	    uint32_t lookahead_head = 0;
	    uint32_t lookahead_count = 0;
	    //nt32_t lookahead_num_4 = LOOKAHEAD_UNITS;
	    uint32_t lookahead_count_4 = 0;
	    uint32_t lookahead_head_4 = 0;
	    uint32_t lookahead_head_mod = 0;
	    uint32_t consumed_bytes = 0;
	    uint32_t consumed_bytes_4 = 0;
	    uint32_t out_bytes = 1;
	    uint32_t cur_header_byte_pos = 0;
	    uint8_t header_byte = 0;
	    uint8_t blocks = 0;
	    uint32_t c = 0;

	    uint32_t used_bytes = 0;
	    const uint8_t* const in_4 = (const uint8_t*) in;

	    while (used_bytes < my_chunk_size) {
		
		//fill up lookahead buffer
		while ((lookahead_count_4 < LOOKAHEAD_UNITS) && (consumed_bytes < my_chunk_size))  {
		    
		    lookahead_4[(lookahead_head_4 + (lookahead_count_4++)) % LOOKAHEAD_UNITS] =
			in_4[in_start_idx + (consumed_bytes_4++)];
		    consumed_bytes+=READ_UNITS;
		    lookahead_count+=READ_UNITS;

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
			hist_head = (hist_head + (HIST_SIZE-hist_count)) % HIST_SIZE;
			hist_count = HIST_SIZE;
		    }
		    header_byte = (header_byte | 1);
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

		    for (size_t i = 0; i < (MIN_MATCH_LENGTH-1); i++) {
			out[out_start_idx + (out_bytes++)] = v & 0x00FF;
			v >>= 8;
		    }
		    uint32_t hist_start = hist_head+hist_count;
		    for (size_t i = 0; i < length; i++) {
			hist[(hist_start+i) % HIST_SIZE] = lookahead[(lookahead_head+i) % HIST_SIZE];
		    }
		    
		    lookahead_head = (lookahead_head + length) % LOOKAHEAD_SIZE;
		    lookahead_count -= length;
		    hist_count += length;
		    if (hist_count > HIST_SIZE) {
			hist_head = (hist_head + (HIST_SIZE-hist_count)) % HIST_SIZE;
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
		else
		    header_byte <<= 1;
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

    __global__ void
    __launch_bounds__(1024, 2)
    kernel_compress(const uint8_t* const in, uint8_t* out, const uint64_t in_n_bytes, const uint64_t out_n_bytes, const uint64_t in_chunk_size, const uint64_t out_chunk_size, const uint64_t n_chunks, uint64_t* lens) {
	uint64_t tid = threadIdx.x + blockDim.x * blockIdx.x;
	compress_func(in, out, in_n_bytes, out_n_bytes, in_chunk_size, out_chunk_size, n_chunks, lens, tid);
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

	uint64_t padded_in_n_bytes = in_n_bytes + (CHUNK_SIZE-(in_n_bytes % CHUNK_SIZE));
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


	cuda_err_chk(cudaMalloc(&d_in, padded_in_n_bytes));
	cuda_err_chk(cudaMalloc(&d_out, len_bytes + exp_data_out_bytes));
	uint64_t* d_lens_out = (uint64_t*) d_out;
	uint8_t* d_data_out = d_out + len_bytes;
	printf("expected_out_bytes: %llu\toverhead_per_chunk: %llu\tchunk_size: %llu\n", exp_data_out_bytes, OVERHEAD_PER_CHUNK_(chunk_size), chunk_size);
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

	printf("in bytes: %llu\n", in_n_bytes);

	*out = new uint8_t[out_bytes];
	uint32_t* out_32 = (uint32_t*) *out;
	out_32[0] = HEAD_INTS ;
	out_32[1] = chunk_size;
	out_32[2] = n_chunks;
	out_32[3] = HIST_SIZE;
	out_32[4] = MIN_MATCH_LENGTH;
	out_32[5] = MAX_MATCH_LENGTH;
	printf("HEAD_INTS: %llu\tchunk_size: %llu\tn_chunks: %llu\tHIST_SIZE: %llu\tMIN_MATCH_LENGTH: %llu\tMAX_MATCH_LENGTH: %llu\tOFFSET_SIZE: %llu\tbitsNeeded(4096): %llu\n",
	       out_32[0], out_32[1], out_32[2], out_32[3], out_32[4], out_32[5], OFFSET_SIZE, bitsNeeded(HIST_SIZE));

	cuda_err_chk(cudaMemcpy((*out) +  head_bytes, d_lens_out, len_bytes, cudaMemcpyDeviceToHost));


	uint64_t out_data_bytes = ((uint64_t*)((*out) + head_bytes))[n_chunks-1];
	printf("out_data_bytes: %llu\n", out_data_bytes);
	
	cuda_err_chk(cudaMemcpy((*out) +  head_bytes + len_bytes, temp, out_data_bytes, cudaMemcpyDeviceToHost));
	*out_n_bytes = head_bytes + len_bytes + out_data_bytes;

	cuda_err_chk(cudaFree(d_out));
	cuda_err_chk(cudaFree(temp));
	
    }

}
