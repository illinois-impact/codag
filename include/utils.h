#ifndef _UTILS_H_
#define _UTILS_H_

#include <cstdint>
#include <utility>

#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <locale.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>

constexpr const char *METADATA_SUFFIX = ".meta";

namespace metadata {
    // struct metadata {
    //     uint64_t out_metadata_num;
    //     uint64_t *out_metadata_lens;
    //     uint64_t **out_metadata;
    // };

    void prepare(std::initializer_list<std::pair<uint64_t, void*>> l,
                    uint64_t& out_metadata_num, 
                    uint64_t* &out_metadata_lens,
                    uint8_t** &out_metadata) {
        auto len = l.size();
        out_metadata_lens = new uint64_t[len];
        out_metadata = new uint8_t*[len];

        auto len_ptr = out_metadata_lens;
        auto data_ptr = out_metadata;
        for (const auto& item: l) {
            *(len_ptr ++) = item.first;
            *(data_ptr ++) = (uint8_t*)item.second;
        }

        out_metadata_num = len;
    }


    void write_to_file(const char *meta_filename,
                    const uint64_t out_metadata_num, uint64_t *out_metadata_lens, uint8_t **out_metadata) {
        char suffixed[128] = { 0 };
        strcpy(suffixed, meta_filename);
        strcat(suffixed, METADATA_SUFFIX);

        std::size_t metadata_size = sizeof(uint64_t) + sizeof(uint64_t) * out_metadata_num;
        for (std::size_t i=0; i<out_metadata_num; ++i) {
            metadata_size += out_metadata_lens[i];
        }


        int fd;
        if((fd = open(suffixed, O_RDWR | O_TRUNC | O_CREAT, S_IRWXU | S_IRGRP | S_IROTH)) == -1) {
            printf("open: %s\n", strerror(errno));
            exit(1);
        }
        if(ftruncate(fd, metadata_size) == -1) {
            printf("ftruncate: %s\n", strerror(errno));
                exit(1);
        }

        uint8_t *map_fd = (uint8_t*)mmap(nullptr, metadata_size, PROT_WRITE | PROT_READ, MAP_SHARED, fd, 0);

        size_t offset = 0;
        memcpy(map_fd + offset, &out_metadata_num, sizeof(uint64_t));
        offset += sizeof(uint64_t);

        memcpy(map_fd + offset, out_metadata_lens, sizeof(uint64_t) * out_metadata_num);
        offset += sizeof(uint64_t) * out_metadata_num;

        for (size_t i=0; i<out_metadata_num; ++i) {
            memcpy(map_fd + offset, out_metadata[i], out_metadata_lens[i]);
            offset += out_metadata_lens[i];
        }

        if(munmap(map_fd, metadata_size) == -1) {
            printf("munmap: %s\n", strerror(errno));
            exit(1);
        }

        close(fd);
    }

    uint8_t* read_from_file(const char *meta_filename,
                    uint64_t &metadata_num, uint64_t* &metadata_lens, uint8_t** &metadata, size_t* map_size) {
        char suffixed[128] = { 0 };
        strcpy(suffixed, meta_filename);
        strcat(suffixed, METADATA_SUFFIX);
        
        int fd;
        if((fd = open(suffixed, O_RDONLY)) == -1) {
            printf("open: %s\n", strerror(errno));
            exit(1);
        }
        struct stat meta_sb;
        fstat(fd, &meta_sb);
        *map_size = meta_sb.st_size;

        uint8_t *map_fd = (uint8_t*)mmap(nullptr, *map_size, PROT_READ, MAP_PRIVATE, fd, 0);
        
        metadata_num = ((uint64_t*) map_fd)[0];
        metadata_lens = (uint64_t*) (map_fd + sizeof(uint64_t));
        metadata = (uint8_t**) malloc(metadata_num * sizeof(uint8_t*));
        uint64_t offset = sizeof(uint64_t) + sizeof(uint64_t) * metadata_num;
        for (size_t i=0; i<metadata_num; ++i) {
            metadata[i] = (uint8_t*) (map_fd + offset);
            offset += metadata_lens[i];
        }

        close(fd);
        return map_fd; // for unmap
    }
}

#endif
