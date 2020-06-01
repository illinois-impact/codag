#ifndef _RLEV2_UTIL_H_
#define _RLEV2_UTIL_H_

#define HEADER_SHORT_REPEAT 0b00000000
#define HEADER_DIRECT       0b01000000
#define HEADER_PACTED_BASE  0b10000000
#define HEADER_DELTA        0b11000000


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


constexpr   uint64_t CHUNK_SIZE_() { return 128 * 1024; }
constexpr   uint16_t BLK_SIZE_() { return 1024; }
constexpr   uint16_t MAX_LITERAL_SIZE_() { return 128; }
constexpr   uint8_t  MINIMUM_REPEAT_() { return 3; }
constexpr   uint8_t  MAXIMUM_REPEAT_() { return 127 + MINIMUM_REPEAT_(); }
constexpr   uint64_t OUTPUT_CHUNK_SIZE_() { return CHUNK_SIZE_() + (CHUNK_SIZE_() - 1) / MAX_LITERAL_SIZE_() + 1; }
constexpr   uint8_t  HIST_LEN_() { return 32; }
constexpr   uint32_t MAX_SHORT_REPEAT_LENGTH_() { return 10; }

#define CHUNK_SIZE                CHUNK_SIZE_()
#define BLK_SIZE                  BLK_SIZE_()			  
#define MAX_LITERAL_SIZE          MAX_LITERAL_SIZE_()
#define MINIMUM_REPEAT            MINIMUM_REPEAT_()
#define MAXIMUM_REPEAT            MAXIMUM_REPEAT_()
#define OUTPUT_CHUNK_SIZE         OUTPUT_CHUNK_SIZE_() // TODO: this is probably not a tight bound.
#define HIST_LEN                  HIST_LEN_()
#define MAX_SHORT_REPEAT_LENGTH   MAX_SHORT_REPEAT_LENGTH_()

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

#endif