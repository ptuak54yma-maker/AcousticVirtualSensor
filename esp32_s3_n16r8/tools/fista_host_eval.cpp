// Host runner for the exact FISTA solver compiled into the ESP32-S3 firmware.
// Input: uint32 frame_count, then frame_count rows of AVS_NUM_BINS float32.
// Output: uint32 frame_count, then frame_count rows of AVS_NUM_COMPONENTS float32.
#include "avs_pipeline.h"
#include "w_standard_mel_data.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "usage: fista_host_eval INPUT.f32 OUTPUT.f32 ITERATIONS\n";
        return 2;
    }
    const unsigned iterations = static_cast<unsigned>(std::stoul(argv[3]));
    std::ifstream input(argv[1], std::ios::binary);
    std::ofstream output(argv[2], std::ios::binary);
    uint32_t frame_count = 0;
    if (!input.read(reinterpret_cast<char *>(&frame_count), sizeof(frame_count))) return 3;
    output.write(reinterpret_cast<const char *>(&frame_count), sizeof(frame_count));
    avs_nnls_workspace_t workspace{};
    if (avs_nnls_init(&workspace, AVS_W_STANDARD) != AVS_OK) return 4;
    std::vector<float> v(AVS_NUM_BINS), h(AVS_NUM_COMPONENTS);
    for (uint32_t frame = 0; frame < frame_count; ++frame) {
        if (!input.read(reinterpret_cast<char *>(v.data()),
                        static_cast<std::streamsize>(v.size() * sizeof(float)))) return 5;
        const avs_status_t status = avs_solve_nnls(
            v.data(), AVS_W_STANDARD, &workspace, h.data(), iterations);
        if (status != AVS_OK) return 6;
        output.write(reinterpret_cast<const char *>(h.data()),
                     static_cast<std::streamsize>(h.size() * sizeof(float)));
    }
    return output ? 0 : 7;
}
