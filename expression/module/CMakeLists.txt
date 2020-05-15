add_library(module SHARED module.cpp)
target_compile_features(module PRIVATE cxx_range_for)
target_link_libraries(module PRIVATE "${TORCH_LIBRARIES}")
