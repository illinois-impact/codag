#ifndef _RLEV2_UTIL_H_
#define _RLEV2_UTIL_H_

#define READ_TYPE int64_t

typedef longlong2 VEC_T;

typedef std::conditional<sizeof(READ_TYPE) == sizeof(int32_t), uint32_t, uint64_t>::type INPUT_T;
typedef std::conditional<sizeof(READ_TYPE) == sizeof(int32_t), uint32_t, uint64_t>::type UINPUT_T;
//typedef std::conditional<sizeof(READ_TYPE) == sizeof(int32_t), longlong2, longlong4>::type VEC_T;

#define ERR_THREAD 0
#define ERR_CHUNK 0

// #define DEBUG

constexpr int WRITE_VEC_SIZE = 4;
constexpr int READ_GRANULARITY = 1;
constexpr int DECODE_UNIT = 4; 
constexpr UINPUT_T VARINT_MASK = 0x7f;

constexpr uint8_t HEADER_MASK         = 0xc0;
constexpr uint8_t HEADER_SHORT_REPEAT = 0b00000000;
constexpr uint8_t HEADER_DIRECT       = 0b01000000;
constexpr uint8_t HEADER_PACTED_BASE  = 0b10000000;
constexpr uint8_t HEADER_DELTA        = 0b11000000;

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

constexpr   uint16_t BLK_SIZE_()                { return (32); }
constexpr   uint64_t CHUNK_SIZE_()              { return (1024 * 8 * 5); }
constexpr   uint32_t INPUT_BUFFER_SIZE()        { return (32); }
constexpr   uint16_t MAX_LITERAL_SIZE_()        { return 128; }
constexpr   uint8_t  MINIMUM_REPEAT_()          { return 3; }
constexpr   uint8_t  MAXIMUM_REPEAT_()          { return 127 + MINIMUM_REPEAT_(); }
// constexpr   uint64_t OUTPUT_CHUNK_SIZE_()       { return CHUNK_SIZE_() + (CHUNK_SIZE_() - 1) / MAX_LITERAL_SIZE_() * 4; }
constexpr   uint64_t OUTPUT_CHUNK_SIZE_()       { return CHUNK_SIZE_() * 2; }
constexpr   uint32_t MAX_SHORT_REPEAT_LENGTH_() { return 10; }
constexpr   uint8_t  HIST_LEN_()                { return 32; }

#define WARP_SIZE                                       32
#define BLK_SIZE                               BLK_SIZE_()			  
//#define CHUNK_SIZE                           CHUNK_SIZE_()
#define INPUT_BUFFER_SIZE              INPUT_BUFFER_SIZE()
#define MAX_LITERAL_SIZE               MAX_LITERAL_SIZE_()
#define MINIMUM_REPEAT                   MINIMUM_REPEAT_()
#define MAXIMUM_REPEAT                   MAXIMUM_REPEAT_()
//#define OUTPUT_CHUNK_SIZE             OUTPUT_CHUNK_SIZE_() // TODO: this is probably not a tight bound.
#define MAX_SHORT_REPEAT_LENGTH MAX_SHORT_REPEAT_LENGTH_()
#define HIST_LEN                               HIST_LEN_()

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

__constant__ uint8_t device_encode_bit_map[65];
__constant__ uint8_t device_closest_bit_map[65];
__constant__ uint8_t device_closest_aligned_bit_map[65];
__constant__ uint8_t device_decode_bit_map[32];

__host__ __device__
inline uint8_t get_encoded_bit_width(const uint8_t& bitwidth) {
#ifdef __CUDA_ARCH__
    return device_encode_bit_map[bitwidth];
#else
    return __BIT_WIDTH_ENCODE_MAP[bitwidth];
#endif
}

__host__ __device__
inline uint8_t get_decoded_bit_width(const uint8_t& bitwidth) {
#ifdef __CUDA_ARCH__
    return device_decode_bit_map[bitwidth];
#else
    return __BIT_WIDTH_DECODE_MAP[bitwidth];
#endif
}

__host__ __device__
inline uint8_t get_closest_bit(const uint8_t bit) {
#ifdef __CUDA_ARCH__
    return device_closest_bit_map[bit];
#else
    return __CLOSEST_FIXED_BIT_MAP[bit];
#endif
}

__host__ __device__
inline uint8_t get_closest_aligned_bit(const uint8_t bit) {
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

constexpr int SINGLE_WARP_DECODE_BUFFER_COUNT = 128;
constexpr int DECODE_BUFFER_COUNT = 32;
constexpr int DECODE_BUFFER4_COUNT = DECODE_BUFFER_COUNT / 4;
constexpr int SHM_BUFFER_COUNT = DECODE_BUFFER_COUNT * BLK_SIZE;

#endif
