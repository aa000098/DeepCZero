#include "deepczero.hpp"

#include <zip.h>
#include <fstream>
#include <vector>
#include <iostream>


int main() {

	const char* filename = "/home/user/project/DeepCZero/models/yolov8n.pt";

    int err = 0;
    zip_t* archive = zip_open(filename, ZIP_RDONLY, &err);
    if (!archive) {
        std::cerr << "Failed to open .pt file as zip (err=" << err << ")\n";
        return 1;
    }

    zip_int64_t num_entries = zip_get_num_entries(archive, 0);
    for (zip_uint64_t i = 0; i < num_entries; ++i) {
        const char* name = zip_get_name(archive, i, 0);
        if (name) {
            std::cout << "Found: " << name << std::endl;
        }
    }

    zip_close(archive);


}
