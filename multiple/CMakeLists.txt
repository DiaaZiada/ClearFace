
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(multiple)
list(APPEND CMAKE_PREFIX_PATH "../libtorch")

find_package(Torch REQUIRED)

add_subdirectory(module)


# Define our library target
add_library(multiple SHARED multiple.cpp)
# Enable C++11
target_compile_features(multiple PRIVATE cxx_range_for)
# Link against LibTorch
target_link_libraries(multiple "${TORCH_LIBRARIES}")
# Link against Multiple
target_link_libraries(multiple module)
