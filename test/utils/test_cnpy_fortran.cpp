#include "cnpy.h"
#include <iostream>
#include <vector>

void test_cnpy_fortran_order() {
    std::cout << "\n=== Testing cnpy F-contiguous array loading ===" << std::endl;

    std::string npz_path = "/home/user/.deepczero/weights/vgg16.npz";
    cnpy::npz_t npz = cnpy::npz_load(npz_path);

    // Load fc6/W
    std::string w_key = "fc6/W";
    auto w_it = npz.find(w_key);

    if (w_it != npz.end()) {
        const cnpy::NpyArray& w_arr = w_it->second;

        std::cout << "\nfc6/W info:" << std::endl;
        std::cout << "  Shape: ";
        for (size_t i = 0; i < w_arr.shape.size(); ++i) {
            std::cout << w_arr.shape[i];
            if (i < w_arr.shape.size() - 1) std::cout << " x ";
        }
        std::cout << std::endl;

        std::cout << "  Fortran order: " << (w_arr.fortran_order ? "true" : "false") << std::endl;
        std::cout << "  Word size: " << w_arr.word_size << " bytes" << std::endl;
        std::cout << "  Num vals: " << w_arr.num_vals << std::endl;

        // Get raw data
        std::vector<float> w_data = w_arr.as_vec<float>();

        std::cout << "\n  First 10 values (raw memory order):" << std::endl;
        std::cout << "    ";
        for (int i = 0; i < 10; ++i) {
            std::cout << w_data[i] << " ";
        }
        std::cout << std::endl;

        // If we interpret as (25088, 4096) in C-order:
        // w_data[0] is at position [0, 0]
        // w_data[1] is at position [0, 1]
        // w_data[4096] is at position [1, 0]

        // But the array was stored as F-order with shape (25088, 4096):
        // This means in the ORIGINAL (Python) array:
        // arr[0, 0] is stored at offset 0
        // arr[1, 0] is stored at offset 1
        // arr[0, 1] is stored at offset 25088

        std::cout << "\n  If interpreted as C-order (25088, 4096):" << std::endl;
        std::cout << "    [0,0] = " << w_data[0] << std::endl;
        std::cout << "    [0,1] = " << w_data[1] << std::endl;
        std::cout << "    [1,0] = " << w_data[4096] << std::endl;

        std::cout << "\n  Actual F-order layout (25088, 4096):" << std::endl;
        std::cout << "    [0,0] = " << w_data[0] << std::endl;
        std::cout << "    [1,0] = " << w_data[1] << std::endl;
        std::cout << "    [0,1] = " << w_data[25088] << std::endl;
    }
}

int main() {
    test_cnpy_fortran_order();
    return 0;
}
