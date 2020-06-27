#ifndef _RLEV2_UTIL_H_
#define _RLEV2_UTIL_H_

#define HEADER_SHORT_REPEAT 0b00000000
#define HEADER_DIRECT       0b01000000
#define HEADER_PACTED_BASE  0b10000000
#define HEADER_DELTA        0b11000000

#include <cstdint>


constexpr uint8_t __BIT_WIDTH_DECODE_MAP[32] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 28, 30, 32, 40, 48, 56, 64 };

constexpr uint8_t __BIT_WIDTH_ENCODE_MAP[65] = { 
    0, 0, 
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
    24, 24, 
    25, 25,
    26, 26, 
    27, 27, 
    28, 28, 28, 28, 28, 28, 28, 28, 
    29, 29, 29, 29, 29, 29, 29, 29, 
    30, 30, 30, 30, 30, 30, 30, 30, 
    31, 31, 31, 31, 31, 31, 31, 31  
}; 

constexpr uint8_t __CLOSEST_FIXED_BIT_MAP[65] = { 
    1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 
    26, 26, 28, 28, 30, 30, 32, 32, 
    40, 40, 40, 40, 40, 40, 40, 40, 
    48, 48, 48, 48, 48, 48, 48, 48, 
    56, 56, 56, 56, 56, 56, 56, 56,
    64, 64, 64, 64, 64, 64, 64, 64 
}; 

constexpr uint8_t __CLOSEST_ALIGNED_FIXED_BIT_MAP[65] = { 
    1, 1, 2, 4, 4, 8, 8, 8, 8, 16, 16, 16, 16, 16, 16, 16, 16, 24, 24, 24, 24, 24, 24, 24, 24, 
    32, 32, 32, 32, 32, 32, 32, 32, 
    40, 40, 40, 40, 40, 40, 40, 40, 
    48, 48, 48, 48, 48, 48, 48, 48,
    56, 56, 56, 56, 56, 56, 56, 56, 
    64, 64, 64, 64, 64, 64, 64, 64 
}; 

constexpr   uint16_t THRDS_SM_() { return (2048); }
constexpr   uint16_t BLK_SIZE_() { return (32); }
constexpr   uint16_t BLKS_SM_()  { return (THRDS_SM_()/BLK_SIZE_()); }
constexpr   uint64_t GRID_SIZE_() { return (1024); }
constexpr   uint64_t NUM_CHUNKS_() { return (GRID_SIZE_()*BLK_SIZE_()); }
constexpr   uint64_t CHUNK_SIZE_() { return (1024*4*32); }
constexpr   uint64_t HEADER_SIZE_() { return (1); }
constexpr   uint32_t HIST_SIZE_() { return 2048; }
constexpr   uint32_t LOOKAHEAD_SIZE_() { return 512; }
constexpr   uint32_t REF_SIZE_() { return 16; }
constexpr   uint32_t REF_SIZE_BYTES_() { return REF_SIZE_()/8; }
constexpr   uint32_t LENGTH_MASK_(uint32_t d) { return ((d > 0) ? 1 | (LENGTH_MASK_(d-1)) << 1 : 0);  }
constexpr   uint8_t DEFAULT_CHAR_() { return ' '; }
constexpr   uint32_t HEAD_INTS_() { return 7; }
constexpr   uint32_t READ_UNITS_() { return 4; }
constexpr   uint32_t LOOKAHEAD_UNITS_() { return LOOKAHEAD_SIZE_()/READ_UNITS_(); }
constexpr   uint64_t WARP_ID_(uint64_t t) { return t/32; }
constexpr   uint32_t LOOKAHEAD_SIZE_4_BYTES_() { return  LOOKAHEAD_SIZE_()/sizeof(uint32_t); }
constexpr   uint32_t HIST_SIZE_4_BYTES_() { return  HIST_SIZE_()/sizeof(uint32_t); }
constexpr   uint32_t INPUT_BUFFER_SIZE() { return (32); }


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
#define MAX_MATCH_LENGTH		  MAX_MATCH_LENGTH_()		  
#define DEFAULT_CHAR			  DEFAULT_CHAR_()			  
#define HEAD_INTS                         HEAD_INTS_()
#define READ_UNITS                        READ_UNITS_()
#define LOOKAHEAD_UNITS                   LOOKAHEAD_UNITS_()
#define WARP_ID(t)                        WARP_ID_(t)
#define INPUT_BUFFER_SIZE                 INPUT_BUFFER_SIZE()

constexpr   uint16_t MAX_LITERAL_SIZE_() { return 128; }
constexpr   uint8_t  MINIMUM_REPEAT_() { return 3; }
constexpr   uint8_t  MAXIMUM_REPEAT_() { return 127 + MINIMUM_REPEAT_(); }
constexpr   uint64_t OUTPUT_CHUNK_SIZE_() { return CHUNK_SIZE_() + (CHUNK_SIZE_() - 1) / MAX_LITERAL_SIZE_() + 1; }
constexpr   uint8_t  HIST_LEN_() { return 32; }
constexpr   uint32_t MAX_SHORT_REPEAT_LENGTH_() { return 10; }

#define MAX_LITERAL_SIZE          MAX_LITERAL_SIZE_()
#define MINIMUM_REPEAT            MINIMUM_REPEAT_()
#define MAXIMUM_REPEAT            MAXIMUM_REPEAT_()
#define OUTPUT_CHUNK_SIZE         OUTPUT_CHUNK_SIZE_() // TODO: this is probably not a tight bound.
#define HIST_LEN                  HIST_LEN_()
#define MAX_SHORT_REPEAT_LENGTH   MAX_SHORT_REPEAT_LENGTH_()

#define ENCODE_UNIT  1 //each thread read 1 unit of input and proceed to next blk
#define DECODE_UNIT  4 //each thread write 4 unit of output and proceed to next blk

typedef uint64_t col_len_t;
typedef uint64_t blk_off_t;

template<typename _Tp>
__host__ __device__
inline const _Tp&
min(const _Tp& __a, const _Tp& __b)
{
    return (__a < __b) ? __a : __b;
}

template<typename _Tp>
__host__ __device__
inline const _Tp&
max(const _Tp& __a, const _Tp& __b)
{
    return (__a > __b) ? __a : __b;
}

template<typename _Tp>
__host__ __device__
inline const _Tp
abs(const _Tp& __x)
{
    return (__x > 0) ? __x : -__x;
}

__constant__ uint8_t device_encode_bit_map[65];
__constant__ uint8_t device_closest_bit_map[65];
__constant__ uint8_t device_closest_aligned_bit_map[65];
__constant__ uint8_t device_decode_bit_map[32];

__host__ __device__
inline uint8_t get_encoded_bit_width(const uint8_t& bitwidth) {
    // static constexpr uint8_t *encode_bit_map = bit_maps;
#ifdef __CUDA_ARCH__
    return device_encode_bit_map[bitwidth];
#else
    return __BIT_WIDTH_ENCODE_MAP[bitwidth];
#endif
}

__host__ __device__
inline uint8_t get_decoded_bit_width(const uint8_t& bitwidth) {
    // static constexpr uint8_t *decode_bit_map = bit_maps + 65 + 65 + 65;
#ifdef __CUDA_ARCH__
    return device_decode_bit_map[bitwidth];
#else
    return __BIT_WIDTH_DECODE_MAP[bitwidth];
#endif
}

__host__ __device__
inline uint8_t get_closest_bit(const uint8_t bit) {
    // static constexpr uint8_t *closest_bit_map = bit_maps + 65;
#ifdef __CUDA_ARCH__
    return device_closest_bit_map[bit];
#else
    return __CLOSEST_FIXED_BIT_MAP[bit];
#endif
}

__host__ __device__
inline uint8_t get_closest_aligned_bit(const uint8_t bit) {
    // static constexpr uint8_t *closest_aligned_bit_map = bit_maps + 65 + 65;
#ifdef __CUDA_ARCH__
    return device_closest_aligned_bit_map[bit];
#else
    return __CLOSEST_ALIGNED_FIXED_BIT_MAP[bit];
#endif
}

inline void initialize_bit_maps() {
    cudaMemcpyToSymbol(device_encode_bit_map, &__BIT_WIDTH_ENCODE_MAP, 65 * sizeof(uint8_t));
    cudaMemcpyToSymbol(device_closest_bit_map, &__CLOSEST_FIXED_BIT_MAP, 65 * sizeof(uint8_t));
    cudaMemcpyToSymbol(device_closest_aligned_bit_map, &__CLOSEST_ALIGNED_FIXED_BIT_MAP, 65 * sizeof(uint8_t));
    cudaMemcpyToSymbol(device_decode_bit_map, &__BIT_WIDTH_DECODE_MAP, 32 * sizeof(uint8_t));
}

#define ERR_THREAD 21

#endif