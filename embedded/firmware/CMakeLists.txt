cmake_minimum_required(VERSION 3.22)
project(avs_stm32f407 LANGUAGES C CXX ASM)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(AVS_CMSIS_DSP_ROOT "${CMAKE_CURRENT_LIST_DIR}/_deps/CMSIS-DSP" CACHE PATH "CMSIS-DSP source root")
set(AVS_CMSIS_CORE_ROOT "${CMAKE_CURRENT_LIST_DIR}/_deps/CMSIS_6/CMSIS/Core" CACHE PATH "CMSIS Core source root")
if(NOT EXISTS "${AVS_CMSIS_DSP_ROOT}/Source/CMakeLists.txt" OR
   NOT EXISTS "${AVS_CMSIS_CORE_ROOT}/Include/cmsis_compiler.h")
    message(FATAL_ERROR "CMSIS dependencies missing. Run powershell -ExecutionPolicy Bypass -File fetch_dependencies.ps1")
endif()

set(CMSISCORE "${AVS_CMSIS_CORE_ROOT}" CACHE PATH "CMSIS Core source root" FORCE)
set(CMSIS_DSP_SOURCE "${AVS_CMSIS_DSP_ROOT}/Source")
set(CMSIS_DSP_INCLUDE "${AVS_CMSIS_DSP_ROOT}/Include")
set(CMSIS_DSP_PRIVATE "${AVS_CMSIS_DSP_ROOT}/PrivateInclude")
set(CMSIS_CORE_INCLUDE "${AVS_CMSIS_CORE_ROOT}/Include")

# Only build the fast real FFT and its CFFT/tables dependencies. The complete
# CMSIS-DSP CMake project contains hundreds of unrelated kernels.
add_library(CMSISDSP STATIC
    "${CMSIS_DSP_SOURCE}/TransformFunctions/arm_rfft_fast_f32.c"
    "${CMSIS_DSP_SOURCE}/TransformFunctions/arm_rfft_fast_init_f32.c"
    "${CMSIS_DSP_SOURCE}/TransformFunctions/arm_cfft_f32.c"
    "${CMSIS_DSP_SOURCE}/TransformFunctions/arm_cfft_init_f32.c"
    "${CMSIS_DSP_SOURCE}/TransformFunctions/arm_cfft_radix8_f32.c"
    "${CMSIS_DSP_SOURCE}/TransformFunctions/arm_bitreversal.c"
    "${CMSIS_DSP_SOURCE}/TransformFunctions/arm_bitreversal2.c"
    "${CMSIS_DSP_SOURCE}/CommonTables/arm_common_tables.c"
    "${CMSIS_DSP_SOURCE}/CommonTables/arm_const_structs.c")
target_include_directories(CMSISDSP PUBLIC "${CMSIS_DSP_INCLUDE}" "${CMSIS_CORE_INCLUDE}")
target_include_directories(CMSISDSP PRIVATE "${CMSIS_DSP_PRIVATE}")
target_compile_definitions(CMSISDSP PUBLIC ARM_MATH_CM4 PRIVATE ARM_MATH_LOOPUNROLL)

set(AVS_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")
add_library(avs_core_stm32 STATIC
    "${AVS_ROOT}/src/avs_pipeline.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/avs_fft_cmsis.cpp"
    "${AVS_ROOT}/src/avs_peak.cpp")
target_include_directories(avs_core_stm32 PUBLIC
    "${AVS_ROOT}/include"
    "${AVS_ROOT}/src"
    "${AVS_CMSIS_DSP_ROOT}/Include"
    "${AVS_CMSIS_CORE_ROOT}/Include")
target_compile_definitions(avs_core_stm32 PRIVATE ARM_MATH_CM4)

add_executable(avs_stm32f407.elf
    "${CMAKE_CURRENT_LIST_DIR}/src/main_firmware.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/startup_stm32f407.S")
target_include_directories(avs_stm32f407.elf PRIVATE
    "${AVS_ROOT}/include"
    "${AVS_ROOT}/src")
target_link_libraries(avs_stm32f407.elf PRIVATE avs_core_stm32 CMSISDSP m)
target_compile_definitions(avs_stm32f407.elf PRIVATE ARM_MATH_CM4)

set(AVS_CPU_FLAGS -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard)
target_compile_options(avs_core_stm32 PRIVATE
    ${AVS_CPU_FLAGS} -ffunction-sections -fdata-sections
    -fno-exceptions -fno-rtti -fno-use-cxa-atexit -fno-threadsafe-statics
    -Os -Wall -Wextra -Wpedantic)
target_compile_options(avs_stm32f407.elf PRIVATE
    ${AVS_CPU_FLAGS} -ffunction-sections -fdata-sections
    -fno-exceptions -fno-rtti -fno-use-cxa-atexit -fno-threadsafe-statics
    -Os -Wall -Wextra -Wpedantic)
target_compile_options(CMSISDSP PRIVATE
    ${AVS_CPU_FLAGS} -ffunction-sections -fdata-sections
    -Os -Wall -Wextra -Wpedantic)

target_link_options(avs_stm32f407.elf PRIVATE
    ${AVS_CPU_FLAGS}
    "--specs=nano.specs" "--specs=nosys.specs"
    "-T${CMAKE_CURRENT_LIST_DIR}/stm32f407vg.ld"
    "-Wl,--gc-sections"
    "-Wl,-Map=${CMAKE_BINARY_DIR}/avs_stm32f407.map"
    "-Wl,--print-memory-usage")
set_target_properties(avs_stm32f407.elf PROPERTIES SUFFIX "")
