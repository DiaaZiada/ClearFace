
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(expression)
list(APPEND CMAKE_PREFIX_PATH "../libtorch")

find_package(Torch REQUIRED)

add_subdirectory(module)


# Define our library target
add_library(expression SHARED expression.cpp)
# Enable C++11
target_compile_features(expression PRIVATE cxx_range_for)
# Link against LibTorch
target_link_libraries(expression "${TORCH_LIBRARIES}")
# Link against Expression
target_link_libraries(expression module)
