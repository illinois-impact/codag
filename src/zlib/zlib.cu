//#include <common.h>
//#include <brle/brle_trans.h>
#include <cstring>
#include <iostream>
#include <sys/types.h>
#include <unistd.h>
#include <zlib/zlib.h>

#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

bool output_check(uint8_t *a, uint64_t a_size, uint8_t *b, uint64_t b_size) {
  if (a_size != b_size)
    return false;

  for (uint64_t i = 0; i < a_size; i++) {
    if (a[i] != b[i])
      return false;
  }
  return true;
}

int main(int argc, char **argv) {
	bool decomp = true;

  if (decomp && (argc < 5)) {
    std::cerr << "Please provide arguments  input output blk_offset chunk_size\n";
    exit(1);
  }

  const char *input = argv[1];
  const char *output =  argv[2];
  const char *blk_f =  argv[3];
  const char *s_input_bytes = argv[4];

  uint64_t chunk_size = (uint64_t)(std::atoi(s_input_bytes));

  const uint16_t col_widths[5] = {32, 64, 128, 256, 512};

  std::chrono::high_resolution_clock::time_point total_start =
      std::chrono::high_resolution_clock::now();
  int in_fd;
  struct stat in_sb;
  void *in;

  int out_fd;
  struct stat out_sb;
  void *out;

  int col_fd;
  struct stat col_sb;
  void *col;

  int blk_fd;
  struct stat blk_sb;
  void *blk;

  if ((in_fd = open(input, O_RDONLY)) == 0) {
    printf("Fatal Error: INPUT File open error\n");
    return -1;
  }
  if ((out_fd = open(output, O_RDWR | O_TRUNC | O_CREAT,
                     S_IRWXU | S_IRGRP | S_IROTH)) == 0) {
    printf("Fatal Error: OUTPUT File open error\n");
    return -1;
  }
  if ((blk_fd = open(blk_f, O_RDONLY)) == 0) {
    printf("Fatal Error: BLK_OFFSET File open error\n");
    return -1;
  }

  fstat(in_fd, &in_sb);

  in = mmap(nullptr, in_sb.st_size, PROT_READ, MAP_PRIVATE, in_fd, 0);

  if (in == (void *)-1) {
    printf("Fatal Error: INPUT Mapping error\n");
    return -1;
  }
  uint8_t *in_ = (uint8_t *)in;
  fstat(blk_fd, &blk_sb);
  blk = mmap(nullptr, blk_sb.st_size, PROT_READ, MAP_PRIVATE, blk_fd, 0);
  if (blk == (void *)-1) {
    printf("Fatal Error: COL_LEN Mapping error\n");
    return -1;
  }
  uint64_t *blk_ = (uint64_t *)blk;

  uint8_t *out_;
  uint64_t out_size;
  std::chrono::high_resolution_clock::time_point compress_start =
      std::chrono::high_resolution_clock::now();
  if (!decomp) {
    std::cerr << "read bytes should be multiple of input bytes \n";
  } else {

    uint8_t *out_2;
    uint64_t out_size2;
    bool out_check = true;
    deflate::decompress_gpu<uint32_t, 262144 / 2, 64, 8, 1>(
        in_, &out_, in_sb.st_size, &out_size,  blk_,
        blk_sb.st_size, chunk_size);
  }

  std::chrono::high_resolution_clock::time_point compress_end =
      std::chrono::high_resolution_clock::now();

  fstat(out_fd, &out_sb);
  if (out_sb.st_size != out_size) {
    if (ftruncate(out_fd, out_size) == -1)
      PRINT_ERROR;
  }
  out = mmap(nullptr, out_size, PROT_WRITE | PROT_READ, MAP_SHARED, out_fd, 0);
  memcpy(out, out_, out_size);

  if (munmap(in, in_sb.st_size) == -1)
    PRINT_ERROR;

  if (munmap(out, out_size) == -1)
    PRINT_ERROR;
  if (munmap(blk, blk_sb.st_size) == -1)
    PRINT_ERROR;

  close(in_fd);
  close(out_fd);
  close(blk_fd);
  free(out_);
  std::chrono::high_resolution_clock::time_point total_end =
      std::chrono::high_resolution_clock::now();
}
