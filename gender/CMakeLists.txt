
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(gender)
list(APPEND CMAKE_PREFIX_PATH "../libtorch")

find_package(Torch REQUIRED)

add_subdirectory(module)


# Define our library target
add_library(gender SHARED gender.cpp)
# Enable C++11
target_compile_features(gender PRIVATE cxx_range_for)
# Link against LibTorch
target_link_libraries(gender "${TORCH_LIBRARIES}")
# Link against Gender
target_link_libraries(gender module)
