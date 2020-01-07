#include <common.h>

constexpr HIST_SIZE = 4096;
constexpr LOOKAHEAD_SIZE = 4096;
constexpr OFFSET_SIZE = (log2<uint32_t>(HIST_SIZE));
constexpr LENGTH_SIZE = (4);
constexpr MIN_MATCH_LENGTH = (ceil<uint32_t>((OFFSET_SIZE+LENGTH_SIZE),8));
constexpr MAX_MATCH_LENGTH = (pow<uint32_t>(2,LENGTH_SIZE) + MIN_MATCH_LENGTH);
constexpr DEFAULT_CHAR  = ' ';

namespace lzss {
    __global__ void compress(const uint8_t* in, uint8_t* out, const uint64_t in_n_bytes, const uint64_t out_n_bytes, const uint64_t in_chunk_size, const uint64_t out_chunk_size, const uint64_t n_chunks, uint64_t* lens) {
	uint64_t tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid < n_chunks) {
	    
	    uint64_t my_chunk_size = (tid == (n_chunks - 1)) ? in_n_bytes % in_chunk_size : in_chunk_size;
	    uint64_t in_start_idx = tid * in_chunk_size;
	    uint64_t out_start_idx = tid * out_chunk_size;

	    uint8_t hist[HIST_SIZE] = {DEFAULT_CHAR};
	    uint8_t lookahead[LOOKAHEAD_SIZE];
	    uint32_t hist_head  = 0;
	    uint32_t hist_count = 0;
	    uint32_t lookahead_head = 0;
	    uint32_t lookahead_count = 0;
	    uint32_t in_head = 0;
	    uint32_t consumed_bytes = 0;
	    uint32_t out_bytes = 1;
	    uint32_t cur_header_byte_pos = 0;
	    uint8_t header_byte = 0;
	    uint8_t blocks = 0;
	    

	    while (consumed_bytes < my_chunk_size) {
	      //fill up lookahead buffer
	      while ((lookahead_count < LOOKAHEAD_SIZE) && (counsumed_bytes < my_chunk_size))  {
		lookahead[(lookahead_head + (lookahead_count++)) % LOOKAHEAD_SIZE] =
		  in[in_start_idx + (consumed_bytes++)];
	      }
		    
		uint32_t offset = 0;
		uint32_t length = 0;
		find_match(hist, hist_head, hist_count, lookahead, lookahead_head, lookahead_count, &offset, &length);
		//no match
		if (length == 0) {
		    uint8_t v = lookahead[lookahead_head]
		    out[out_start_idx + (out_bytes++)] = v;
		    lookahead_head = (lookahead_head + 1) % LOOKAHEAD_SIZE;
		    lookahead_count--;
		    hist[hist_head] = v;
		    hist_head = (hist_head+1) % HIST_SIZE;
		    hist_count = (hist_count == HIST_SIZE) ? hist_count : (hist_count+1);
		    header_byte = (header_byte | 1);
		    
		    
		    

		}
		//match
		else {
		    uint64_t v = (offset << LENGTH_SIZE) | (length - MIN_MATCH_LENGTH);

		    for (size_t i = 0; i < MIN_MATCH_LENGTH; i++) {
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
		      hist_count = HIST_SIZE:
		    }
		    
		    //int offset_start = mod(hist_count - offset, HIST_SIZE);

		    
		    //header_byte = (header_byte << 1);
		}
		if ((++blocks) == 8) {
		    out[out_start_idx + cur_header_byte_pos] = header_byte;
		    header_byte = 0;
		    blocks = 0;
		    cur_header_byte_pos = out_bytes++;
		}
		else
		  header_byte <<= 1;
	    }
	    if (blocks != 0) {
		out[out_start_idx + cur_header_byte_pos] = header_byte;

	    }
	    lens[tid] = out_bytes;
	    if (out_bytes > my_chunk_size)
		printf("comrpessed alrger than uncompressed\n");

	}
	
    }

}
