set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR cortex-m4)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(CMAKE_C_COMPILER arm-none-eabi-gcc CACHE FILEPATH "Arm GNU C compiler")
set(CMAKE_CXX_COMPILER arm-none-eabi-g++ CACHE FILEPATH "Arm GNU C++ compiler")
set(CMAKE_ASM_COMPILER arm-none-eabi-gcc CACHE FILEPATH "Arm GNU assembler driver")
set(CMAKE_OBJCOPY arm-none-eabi-objcopy CACHE FILEPATH "Arm GNU objcopy")
set(CMAKE_SIZE arm-none-eabi-size CACHE FILEPATH "Arm GNU size")
