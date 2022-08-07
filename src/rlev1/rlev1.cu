#include <common.h>
#include <cstring>
#include <iostream>
#include <rlev1/rlev1.h>
#include <sys/types.h>
#include <unistd.h>

#include <chrono>
#include <fcntl.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>

using namespace std;

//#define LOG_ALL(f_, ...) printf((f_), ##__VA_ARGS__)

void comp_test(uint64_t num, uint8_t *in, uint8_t *out) {
  for (int i = 0; i < num; i++) {
    if (in[i] != out[i]) {
      std::cout << "File Differ at " << i << std::endl;
      break;
    }
  }
}

__host__ void print_usage() {
  printf("  %-35s Input binary filename (required).\n", "-f, --file");
  printf("  %-35s Output binary filename (required).\n", "-o, --output");
  printf("  %-35s Reading data type size in bytes. Reqading data type size "
         "should be a multiple of Data type size (required).\n",
         "-r, --read_byte");
  printf("  %-35s Data type size in bytes (required).\n", "-t, --data_byte");
  printf("  %-35s Length of column before compression (required).\n",
         "-l, --comp_len");
  printf("  %-35s Decompression option (true for decompression, default "
         "compression).\n",
         "-d, --decomp");

  exit(1);
}

int main(int argc, char **argv) {
  string fname = "";
  string output_fname = "";
  bool decomp = false;

  int input_bytes = 0;
  int read_bytes = 0;
  int COMP_COL_LEN = 32 * 1;

  int num_warp = 1;

  while (1) {
    int option_index = 0;
    static struct option long_options[]{
        {"file", required_argument, 0, 'f'},
        //{"output", required_argument, 0, 'o'},
        //{"read_byte", required_argument, 0, 'r'},
        {"data_byte", required_argument, 0, 't'},
        {"num_warp", no_argument, 0, 'n'},
        {"decomp", no_argument, 0, 'd'},
        //{"comp_len", required_argument, 0, 'l'}
    };
    int c;

    c = getopt_long(argc, argv, "f:o:r:t:d:l:n:?", long_options, &option_index);
    if (c == -1)
      break;

    switch (c) {
    case 'f':
      fname = optarg;
      break;

    case 'o':
      output_fname = optarg;
      break;

    case 'd':
      if (strcmp(optarg, "true") == 0)
        decomp = true;
      else
        decomp = false;
      break;

    case 'r':
      read_bytes = atoi(optarg);
      break;

    case 't':
      input_bytes = atoi(optarg);
      break;

    case 'l':
      COMP_COL_LEN = atoi(optarg);
      break;
    case 'n':
      num_warp = atoi(optarg);
      break;

    default:
      print_usage();
      break;
    }
  }

  // const char* input = decomp ? argv[2] : argv[1];
  // const char* output = decomp ? argv[3] : argv[2];
  // const char* s_input_bytes = decomp ? argv[4] : argv[3];
  // const char* s_read_bytes = decomp ? argv[5] : argv[4];

  const char *input = fname.c_str();
  const char *output = output_fname.c_str();

  // int input_bytes = std::atoi(s_input_bytes);
  // int read_bytes = std::atoi(s_read_bytes);

  // std::cout << "read bytes: " << read_bytes << std::endl;

  std::chrono::high_resolution_clock::time_point total_start =
      std::chrono::high_resolution_clock::now();
  int in_fd;
  struct stat in_sb;
  void *in;

  int out_fd;
  struct stat out_sb;
  void *out;

  if ((in_fd = open(input, O_RDONLY)) == 0) {
    printf("Fatal Error: INPUT File open error\n");
    return -1;
  }
  if ((out_fd = open(output, O_RDWR | O_TRUNC | O_CREAT,
                     S_IRWXU | S_IRGRP | S_IROTH)) == 0) {
    printf("Fatal Error: OUTPUT File open error\n");
    return -1;
  }

  fstat(in_fd, &in_sb);

  in = mmap(nullptr, in_sb.st_size, PROT_READ, MAP_PRIVATE, in_fd, 0);

  if (in == (void *)-1) {
    printf("Fatal Error: INPUT Mapping error\n");
    return -1;
  }
  uint8_t *in_ = (uint8_t *)in;
  uint8_t *out_;
  uint64_t out_size;

  std::cout << "CHUNK_SIZE"
            << "\t"
            << "COLUMN WIDTH\t BUFFER SIZE \t COMPRESSED SIZE \t META DATA "
               "SIZE \t DECOMPRESSION TIME\n";

  if (num_warp == 1) {

    if ((input_bytes) == 1) {
      uint8_t *out_2;
      uint64_t out_size2;
      uint64_t chunk_size = 131072;
      int COMP_COL_LEN_Test = 8;
      rle_v1::compress_gpu_orig<uint64_t, uint8_t, uint32_t>(
          in_, &out_, in_sb.st_size, &out_size, COMP_COL_LEN_Test, chunk_size);
      rle_v1::decompress_gpu_orig<uint32_t, uint8_t, uint32_t, 64, 1, 1>(
          out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      comp_test(out_size2, in_, out_2);
      free(out_2);

      rle_v1::decompress_gpu_orig<uint32_t, uint8_t, uint32_t, 64, 2, 1>(
          out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      comp_test(out_size2, in_, out_2);
      free(out_2);

      rle_v1::decompress_gpu_orig<uint32_t, uint8_t, uint32_t, 64, 4, 1>(
          out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      comp_test(out_size2, in_, out_2);
      free(out_2);

      rle_v1::decompress_gpu_orig<uint32_t, uint8_t, uint32_t, 64, 8, 1>(
          out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      comp_test(out_size2, in_, out_2);
      free(out_2);

      rle_v1::decompress_gpu_orig<uint32_t, uint8_t, uint32_t, 64, 16, 1>(
          out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      comp_test(out_size2, in_, out_2);
      free(out_2);
    }

    else if ((input_bytes) == 2) {
      uint8_t *out_2;
      uint64_t out_size2;
      uint64_t chunk_size = 131072;
      int COMP_COL_LEN_Test = 8;
      rle_v1::compress_gpu_orig<uint64_t, uint16_t, uint32_t>(
          in_, &out_, in_sb.st_size, &out_size, COMP_COL_LEN_Test, chunk_size);
      rle_v1::decompress_gpu_orig<uint32_t, uint16_t, uint32_t, 64, 1, 1>(
          out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      comp_test(out_size2, in_, out_2);
      free(out_2);

      rle_v1::decompress_gpu_orig<uint32_t, uint16_t, uint32_t, 64, 2, 1>(
          out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      comp_test(out_size2, in_, out_2);
      free(out_2);

      rle_v1::decompress_gpu_orig<uint32_t, uint16_t, uint32_t, 64, 4, 1>(
          out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      comp_test(out_size2, in_, out_2);
      free(out_2);

      rle_v1::decompress_gpu_orig<uint32_t, uint16_t, uint32_t, 64, 8, 1>(
          out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      comp_test(out_size2, in_, out_2);
      free(out_2);

      rle_v1::decompress_gpu_orig<uint32_t, uint16_t, uint32_t, 64, 16, 1>(
          out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      comp_test(out_size2, in_, out_2);
      free(out_2);
    }

    else if ((input_bytes) == 4) {
      uint8_t *out_2;
      uint64_t out_size2;
      uint64_t chunk_size = 131072;
      int COMP_COL_LEN_Test = 8;
      rle_v1::compress_gpu_orig<uint64_t, uint32_t, uint32_t>(
          in_, &out_, in_sb.st_size, &out_size, COMP_COL_LEN_Test, chunk_size);
      rle_v1::decompress_gpu_orig<uint32_t, uint32_t, uint32_t, 64, 1, 1>(
          out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      comp_test(out_size2, in_, out_2);
      free(out_2);

      rle_v1::decompress_gpu_orig<uint32_t, uint32_t, uint32_t, 64, 2, 1>(
          out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      comp_test(out_size2, in_, out_2);
      free(out_2);

      rle_v1::decompress_gpu_orig<uint32_t, uint32_t, uint32_t, 64, 4, 1>(
          out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      comp_test(out_size2, in_, out_2);
      free(out_2);

      rle_v1::decompress_gpu_orig<uint32_t, uint32_t, uint32_t, 64, 8, 1>(
          out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      comp_test(out_size2, in_, out_2);
      free(out_2);

      rle_v1::decompress_gpu_orig<uint32_t, uint32_t, uint32_t, 64, 16, 1>(
          out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      comp_test(out_size2, in_, out_2);
      free(out_2);
    }

    else if ((input_bytes) == 8) {
      uint8_t *out_2;
      uint64_t out_size2;
      uint64_t chunk_size = 131072;
      //            chunk_size = 10240;

      int COMP_COL_LEN_Test = 8;

      rle_v1::compress_gpu_orig<uint64_t, uint64_t, uint32_t>(
          in_, &out_, in_sb.st_size, &out_size, COMP_COL_LEN_Test, chunk_size);
      // printf("comp data: %lx\n", ((uint32_t*)out_)[0+1]);
      // printf("comp data: %lx\n", ((uint32_t*)out_)[32+1]);

      rle_v1::decompress_gpu_orig<uint32_t, uint64_t, uint32_t, 64, 1, 1>(
          out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      comp_test(out_size2, in_, out_2);
      free(out_2);

      // rle_v1::decompress_gpu_orig<uint32_t, uint64_t, uint32_t, 64,2,
      // 1>(out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      // comp_test(out_size2, in_, out_2);
      // free(out_2);

      // rle_v1::decompress_gpu_orig<uint32_t, uint64_t, uint32_t, 64,4,
      // 1>(out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      // comp_test(out_size2, in_, out_2);
      // free(out_2);

      // rle_v1::decompress_gpu_orig<uint32_t, uint64_t, uint32_t, 64,8,
      // 1>(out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      // comp_test(out_size2, in_, out_2);
      // free(out_2);

      // rle_v1::decompress_gpu_orig<uint32_t, uint64_t, uint32_t, 64,16,
      // 1>(out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      // comp_test(out_size2, in_, out_2);
      // free(out_2);

    }

    else {
      printf("Not supported input bytes\n");
    }
  }

  else if (num_warp == 2) {

    if ((input_bytes) == 1) {
      uint8_t *out_2;
      uint64_t out_size2;
      uint64_t chunk_size = 131072;
      int COMP_COL_LEN_Test = 8;
      rle_v1::compress_gpu_orig<uint64_t, uint8_t, uint32_t>(
          in_, &out_, in_sb.st_size, &out_size, COMP_COL_LEN_Test, chunk_size);
      rle_v1::decompress_gpu_orig<uint32_t, uint8_t, uint32_t, 64, 1>(
          out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      comp_test(out_size2, in_, out_2);
      free(out_2);

      rle_v1::decompress_gpu_orig<uint32_t, uint8_t, uint32_t, 64, 2>(
          out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      comp_test(out_size2, in_, out_2);
      free(out_2);

      rle_v1::decompress_gpu_orig<uint32_t, uint8_t, uint32_t, 64, 4>(
          out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      comp_test(out_size2, in_, out_2);
      free(out_2);

      rle_v1::decompress_gpu_orig<uint32_t, uint8_t, uint32_t, 64, 8>(
          out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      comp_test(out_size2, in_, out_2);
      free(out_2);

      rle_v1::decompress_gpu_orig<uint32_t, uint8_t, uint32_t, 64, 16>(
          out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      comp_test(out_size2, in_, out_2);
      free(out_2);
    }

    else if ((input_bytes) == 2) {
      uint8_t *out_2;
      uint64_t out_size2;
      uint64_t chunk_size = 131072;
      int COMP_COL_LEN_Test = 8;
      rle_v1::compress_gpu_orig<uint64_t, uint16_t, uint32_t>(
          in_, &out_, in_sb.st_size, &out_size, COMP_COL_LEN_Test, chunk_size);
      rle_v1::decompress_gpu_orig<uint32_t, uint16_t, uint32_t, 64, 1>(
          out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      comp_test(out_size2, in_, out_2);
      free(out_2);

      rle_v1::decompress_gpu_orig<uint32_t, uint16_t, uint32_t, 64, 2>(
          out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      comp_test(out_size2, in_, out_2);
      free(out_2);

      rle_v1::decompress_gpu_orig<uint32_t, uint16_t, uint32_t, 64, 4>(
          out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      comp_test(out_size2, in_, out_2);
      free(out_2);

      rle_v1::decompress_gpu_orig<uint32_t, uint16_t, uint32_t, 64, 8>(
          out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      comp_test(out_size2, in_, out_2);
      free(out_2);

      rle_v1::decompress_gpu_orig<uint32_t, uint16_t, uint32_t, 64, 16>(
          out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      comp_test(out_size2, in_, out_2);
      free(out_2);
    }

    else if ((input_bytes) == 4) {
      uint8_t *out_2;
      uint64_t out_size2;
      uint64_t chunk_size = 131072;
      int COMP_COL_LEN_Test = 8;
      rle_v1::compress_gpu_orig<uint64_t, uint32_t, uint32_t>(
          in_, &out_, in_sb.st_size, &out_size, COMP_COL_LEN_Test, chunk_size);
      rle_v1::decompress_gpu_orig<uint32_t, uint32_t, uint32_t, 64, 1>(
          out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      comp_test(out_size2, in_, out_2);
      free(out_2);

      rle_v1::decompress_gpu_orig<uint32_t, uint32_t, uint32_t, 64, 2>(
          out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      comp_test(out_size2, in_, out_2);
      free(out_2);

      rle_v1::decompress_gpu_orig<uint32_t, uint32_t, uint32_t, 64, 4>(
          out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      comp_test(out_size2, in_, out_2);
      free(out_2);

      rle_v1::decompress_gpu_orig<uint32_t, uint32_t, uint32_t, 64, 8>(
          out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      comp_test(out_size2, in_, out_2);
      free(out_2);

      rle_v1::decompress_gpu_orig<uint32_t, uint32_t, uint32_t, 64, 16>(
          out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      comp_test(out_size2, in_, out_2);
      free(out_2);
    }

    else if ((input_bytes) == 8) {
      uint8_t *out_2;
      uint64_t out_size2;
      uint64_t chunk_size = 131072;
      int COMP_COL_LEN_Test = 8;

      rle_v1::compress_gpu_orig<uint64_t, uint64_t, uint32_t>(
          in_, &out_, in_sb.st_size, &out_size, COMP_COL_LEN_Test, chunk_size);
      rle_v1::decompress_gpu_orig<uint32_t, uint64_t, uint32_t, 64, 1>(
          out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      comp_test(out_size2, in_, out_2);
      free(out_2);

      rle_v1::decompress_gpu_orig<uint32_t, uint64_t, uint32_t, 64, 2>(
          out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      comp_test(out_size2, in_, out_2);
      free(out_2);

      rle_v1::decompress_gpu_orig<uint32_t, uint64_t, uint32_t, 64, 4>(
          out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      comp_test(out_size2, in_, out_2);
      free(out_2);

      rle_v1::decompress_gpu_orig<uint32_t, uint64_t, uint32_t, 64, 8>(
          out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      comp_test(out_size2, in_, out_2);
      free(out_2);

      rle_v1::decompress_gpu_orig<uint32_t, uint64_t, uint32_t, 64, 16>(
          out_, &out_2, out_size, &out_size2, COMP_COL_LEN_Test, chunk_size);
      comp_test(out_size2, in_, out_2);
      free(out_2);

    }

    else {
      printf("Not supported input bytes\n");
    }
  }

  if (munmap(in, in_sb.st_size) == -1)
    PRINT_ERROR;
  // if(munmap(out, out_size) == -1) PRINT_ERROR;

  close(in_fd);
}
