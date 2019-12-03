#ifndef BINARIZE_H
#define BINARIZE_H

#include <string>
#include <fstream>
#include <cstdint>

enum type { uint8,
	    uint16,
	    uint32,
	    uint64,
	    int8,
	    int16,
	    int32,
	    int64,
	    timestamp,
	    str,
	    sp,
	    dp,
          };

const std::string empty_str = "\0";

void binarize(const std::string& in_file, const std::string& out_file, type t) {
    std::ifstream ifs(in_file, std::ifstream::in);
    std::ofstream ofs(out_file, std::ofstream::out | std::ofstream::binary);

    std::string line;
    while(std::getline(ifs, line)) {
        bool val_exists = line.length() != 0;
	switch (t) {
	case uint8: {
	    uint8_t v = val_exists ? stoull(line) : 0;
	    ofs.write((char*) &v, sizeof(uint8_t));
	    break;
	}
	case uint16: {
	    uint16_t v = val_exists ? stoull(line) : 0;
	    ofs.write((char*) &v, sizeof(uint16_t));
	    break;
	}
	case uint32: {
	    uint32_t v = val_exists ? stoull(line) : 0;
	    ofs.write((char*) &v, sizeof(uint32_t));
	    break;
	}
	case uint64: {
	    uint64_t v = val_exists ? stoull(line) : 0;
	    ofs.write((char*) &v, sizeof(uint64_t));
	    break;
	}
	case int8: {
	    int8_t v = val_exists ? stoll(line) : 0;
	    ofs.write((char*) &v, sizeof(int8_t));
	    break;
	}
	case int16: {
	    int16_t v = val_exists ? stoll(line) : 0;
	    ofs.write((char*) &v, sizeof(int16_t));
	    break;
	}
	case int32: {
	    int32_t v = val_exists ? stoll(line) : 0;
	    ofs.write((char*) &v, sizeof(int32_t));
	    break;
	}
	case int64: {
	    int64_t v = val_exists ? stoll(line) : 0;
	    ofs.write((char*) &v, sizeof(int64_t));
	    break;
	}
	case timestamp: {
	    uint64_t v = val_exists ? stoull(line) : 0;
	    ofs.write((char*) &v, sizeof(uint64_t));
	    break;
	}
	case str: {
	    std::string v = val_exists ? line + "\0" : empty_str;
	    ofs.write((char*) v.c_str(), v.length());
	    break;
	}
	case sp: {
	    float v = val_exists ? stof(line) : 0;
	    ofs.write((char*) &v, sizeof(float));
	    break;
	}
	case dp: {
	    double v = val_exists ? stod(line) : 0;
	    ofs.write((char*) &v, sizeof(double));
	    break;
	}
	default: {
	    break;
	}

	}
	
    }
    ifs.close();
    ofs.close();
}

#endif
