#include <base_deflate.h>
#include <iostream>


static void show_usage(const char* name) {
    std::cerr << "Usage: " << name << " SOURCE_FILE DESTINATION_FILE"
              << std::endl;
}


int main(int argc, char* argv[]) {
    if (argc != 3) {
        show_usage(argv[0]);
        return 1;
    }

    const std::string src_file = argv[1];
    const std::string dst_file = argv[2];

    uncompress(src_file, dst_file);

}
