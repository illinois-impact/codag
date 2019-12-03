#include <binarize.h>
#include <iostream>


static void show_usage(const char* name) {
    std::cerr << "Usage: " << name << " SOURCE_FILE DESTINATION_FILE TYPE"
              << "\n\n\tTYPE:\n"
              << "\t\t0\tuint8\n"
              << "\t\t1\tuint16\n"
	      << "\t\t2\tuint32\n"
	      << "\t\t3\tuint64\n"
	      << "\t\t4\tint8\n"
	      << "\t\t5\tint16\n"
	      << "\t\t6\tint32\n"
	      << "\t\t7\tint64\n"
	      << "\t\t8\timestamp\n"
	      << "\t\t8\tstr\n"
	      << "\t\t10\tsp\n"
	      << "\t\t11\tdp\n"
              << std::endl;
}


int main(int argc, char* argv[]) {
    if (argc != 4) {
	show_usage(argv[0]);
        return 1;
    }

    const std::string src_file = argv[1];
    const std::string dst_file = argv[2];
    const std::string t_str = argv[3];
    const unsigned int t = std::stoul(t_str);

    if (t > dp) {
	show_usage(argv[0]);
        return 1;
    }

    binarize(src_file, dst_file, (type)t);
	 

}
