#include <common.h>
#include <rlev2/rlev2.h>

#include <cassert>

void test(int64_t ll[], uint8_t exp[], int64_t in_n_bytes, int64_t out_exp_bytes) {
    uint8_t *out;
    uint64_t out_size;

    rlev2::compress_gpu(ll, in_n_bytes, out, out_size);
    uint16_t chunks = *((uint16_t*)out);
    const auto base = sizeof(uint32_t) + chunks * sizeof(uint64_t) + sizeof(uint64_t);

    for (int i=0; i<out_exp_bytes; ++i) {
        // printf("out[%d]: %x\n", i, out[base + i]);
        assert(exp[i] == out[base + i]);
    }
    delete[] out;
}

void test_PB() {
    int64_t ll[] =  {2030, 2000, 2020, 1000000, 2040, 2050, 2060, 2070, 2080, 2090, 2100, 2110, 2120, 2130, 2140, 2150, 2160, 2170, 2180, 2190};
    uint8_t exp[] = {0x8e, 0x13, 0x2b, 0x21, 0x07, 0xd0, 0x1e, 0x00, 0x14, 0x70, 0x28, 0x32, 0x3c, 0x46, 0x50, 0x5a, 0x64, 0x6e, 0x78, 0x82, 0x8c, 0x96, 0xa0, 0xaa, 0xb4, 0xbe, 0xfc, 0xe8};
    test(ll, exp, sizeof(ll), sizeof(exp));

    fprintf(stderr, "====== DIRECTE PASSED =====\n");
}

void test_DIRECT() {
    int64_t ll[] =  {23713, 43806, 57005, 48879};
    uint8_t exp[] = {0x5e, 0x03, 0x5c, 0xa1, 0xab, 0x1e, 0xde, 0xad, 0xbe, 0xef};

    test(ll, exp, sizeof(ll), sizeof(exp));

    fprintf(stderr, "====== DIRECTE PASSED =====\n");
}

void test_DELTA() {
    int64_t ll[] =  {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};
    uint8_t exp[] = {0xc6, 0x09, 0x02, 0x02, 0x22, 0x42, 0x42, 0x46};
   
    test(ll, exp, sizeof(ll), sizeof(exp));

    fprintf(stderr, "====== DELTA PASSED =====\n");
}

void test_SHORTREPEAT() {
    int64_t ll[] =  {10000, 10000, 10000, 10000, 10000};
    uint8_t exp[] = {0x0a, 0x27, 0x10};
    
    test(ll, exp, sizeof(ll), sizeof(exp));

    fprintf(stderr, "====== SHORT REPEAT PASSED =====\n");
}

int main() {
    test_DIRECT();
    test_SHORTREPEAT();
    test_DIRECT();
    test_PB();
    return 0;
}
