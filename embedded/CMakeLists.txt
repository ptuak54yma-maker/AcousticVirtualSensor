cmake_minimum_required(VERSION 3.20)
project(avs_embedded_stage1 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(avs_core
    src/avs_pipeline.cpp
    src/avs_fft.cpp
    src/avs_peak.cpp
)
target_include_directories(avs_core PUBLIC include src)

if(MSVC)
    target_compile_options(avs_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(avs_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(avs_smoke_test tests/smoke_test.cpp)
target_link_libraries(avs_smoke_test PRIVATE avs_core)

# The real model array is generated from the existing W_standard.npy by the
# dictionary exporter. Keep smoke tests buildable before generating the header.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/w_standard_data.h")
    add_executable(avs_infer src/main.cpp)
    target_link_libraries(avs_infer PRIVATE avs_core)
endif()

enable_testing()
add_test(NAME avs_smoke_test COMMAND avs_smoke_test)
