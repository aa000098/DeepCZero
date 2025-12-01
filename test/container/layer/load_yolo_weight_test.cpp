#include "deepczero.hpp"

#include <zip.h>
#include <fstream>
#include <vector>
#include <iostream>

void load_yolo_weight_test() {

	std::string filename = "/home/user/project/DeepCZero/models/yolov8n.pt";
    std::string pkl_output_path = "/home/user/project/DeepCZero/models/yolov8n/data.pkl";  // 저장 위치

    int err = 0;
    zip_t* archive = zip_open(filename.c_str(), ZIP_RDONLY, &err);
    if (!archive) {
        std::cerr << "Failed to open .pt file as zip (err=" << err << ")\n";
        return 1;
    }


    std::string target_file = "yolov8n/data.pkl";
    zip_stat_t st;
    if (zip_stat(archive, target_file.c_str(), 0, &st) != 0) {
        std::cerr << "Failed to stat: " << target_file << "\n";
        zip_close(archive);
        return 1;
    }

    zip_file_t* zf = zip_fopen(archive, target_file.c_str(), 0);
    if (!zf) {
        std::cerr << "Failed to open: " << target_file << "\n";
        zip_close(archive);
        return 1;
    }

    std::vector<char> buffer(st.size);
    zip_fread(zf, buffer.data(), st.size);
    zip_fclose(zf);
   // zip_close(archive);

    // 출력: 바이너리 파일 저장
    std::ofstream fout(pkl_output_path, std::ios::binary);
    fout.write(buffer.data(), buffer.size());
    fout.close();

    std::cout << "Extracted " << target_file << " to " << pkl_output_path << " (" << buffer.size() << " bytes)\n";

    // 출력: 일부 HEX 출력
    for (size_t i = 0; i < std::min(buffer.size(), size_t(64)); ++i) {
        printf("%02X ", static_cast<unsigned char>(buffer[i]));
        if ((i + 1) % 16 == 0) std::cout << "\n";
    }


    zip_int64_t num_entries = zip_get_num_entries(archive, 0);
    for (zip_int64_t i = 0; i < num_entries; ++i) {
        const char* name = zip_get_name(archive, i, 0);
        if (name) {
            std::cout << "Found: " << name << std::endl;
        }
    }

    zip_close(archive);


}

int main() {
	load_yolo_weight_test();	
}
