cmake_minimum_required(VERSION 3.10)
project(OptimaKmeans LANGUAGES C CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(CUDAToolkit REQUIRED)

add_subdirectory(src)
add_executable(example main.c)
target_link_libraries(example optima_kmeans m)

add_executable(autotune autotune.c)
target_link_libraries(autotune optima_kmeans m)