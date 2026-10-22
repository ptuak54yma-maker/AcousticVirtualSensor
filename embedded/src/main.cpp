#include "avs_pipeline.h"
#include "w_standard_data.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: avs_infer <mono_float32_raw.bin> <output.csv>\n";
        return 2;
    }

    const std::string input_path = argv[1];
    const std::string output_path = argv[2];
    std::ifstream input(input_path, std::ios::binary);
    if (!input) {
        std::cerr << "Cannot open input: " << input_path << "\n";
        return 3;
    }
    std::vector<char> bytes((std::istreambuf_iterator<char>(input)),
                            std::istreambuf_iterator<char>());
    if (bytes.size() % sizeof(float) != 0) {
        std::cerr << "Input byte count is not a multiple of float32.\n";
        return 4;
    }
    std::vector<float> samples(bytes.size() / sizeof(float));
    if (!bytes.empty()) std::memcpy(samples.data(), bytes.data(), bytes.size());
    if (samples.size() < AVS_N_FFT) {
        std::cerr << "Input must contain at least " << AVS_N_FFT << " float samples.\n";
        return 5;
    }

    std::vector<float> window(AVS_N_FFT);
    avs_make_hann(window.data(), window.size());
    avs_nnls_workspace_t workspace{};
    if (avs_nnls_init(&workspace, AVS_W_STANDARD) != AVS_OK) {
        std::cerr << "Could not initialize NNLS workspace.\n";
        return 6;
    }

    std::ofstream output(output_path);
    if (!output) {
        std::cerr << "Cannot write output: " << output_path << "\n";
        return 7;
    }
    output << "frame,h_event_sum";
    for (size_t j = 0; j < AVS_NUM_COMPONENTS; ++j) output << ",h" << j;
    output << "\n";
    output.precision(9);

    avs_spectrum_t spectrum{};
    float h[AVS_NUM_COMPONENTS] = {};
    std::vector<float> envelope;
    const size_t num_frames = (samples.size() - AVS_N_FFT) / AVS_HOP_LENGTH + 1;

    for (size_t frame_index = 0; frame_index < num_frames; ++frame_index) {
        const size_t start = frame_index * AVS_HOP_LENGTH;
        float mean = 0.0f;
        for (size_t i = 0; i < AVS_N_FFT; ++i) mean += samples[start + i];
        mean /= static_cast<float>(AVS_N_FFT);

        float frame[AVS_N_FFT];
        for (size_t i = 0; i < AVS_N_FFT; ++i) frame[i] = samples[start + i] - mean;

        const auto fft_status = avs_compute_spectrum(frame, window.data(), &spectrum);
        if (fft_status != AVS_OK) {
            std::cerr << "Spectrum stage failed at frame " << frame_index << "\n";
            return 8;
        }
        const auto nnls_status = avs_solve_nnls(spectrum.log_magnitude, AVS_W_STANDARD,
                                                 &workspace, h, 20000, 1e-6f);
        if (nnls_status != AVS_OK) {
            std::cerr << "NNLS failed at frame " << frame_index
                      << " with status " << static_cast<int>(nnls_status) << "\n";
            return 9;
        }
        const float h_event = avs_sum_event_activations(h, AVS_NUM_COMPONENTS);
        envelope.push_back(h_event);
        output << frame_index << "," << h_event;
        for (float x : h) output << "," << x;
        output << "\n";
    }
    output.close();

    const std::string peaks_path = output_path + ".peaks.csv";
    std::ofstream peaks_file(peaks_path);
    if (!peaks_file) {
        std::cerr << "Cannot write peaks output: " << peaks_path << "\n";
        return 10;
    }
    size_t max_peaks = envelope.size();
    std::vector<size_t> peak_indices(max_peaks);
    const size_t peak_count = avs_detect_peaks(envelope.data(), envelope.size(),
                                                AVS_PEAK_PROMINENCE, AVS_PEAK_DISTANCE,
                                                peak_indices.data(), max_peaks);
    peaks_file << "peak_frame\n";
    for (size_t i = 0; i < peak_count; ++i) peaks_file << peak_indices[i] << "\n";

    std::cout << "Frames: " << num_frames << "\n";
    std::cout << "Peak count (C++ implementation): " << peak_count << "\n";
    std::cout << "Activation CSV: " << output_path << "\n";
    std::cout << "Peak CSV: " << peaks_path << "\n";
    std::cout << "Native execution validates logic only; it is not STM32 timing evidence.\n";
    return 0;
}
