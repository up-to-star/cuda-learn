cmake_minimum_required(VERSION 3.20)

project(cuda-learn VERSION 0.1.0 LANGUAGES CUDA CXX C)

find_package(CUDAToolkit)

# set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
# # 让 nvcc 生成 clangd 能解析的编译命令
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -x cu --compiler-options '-fPIC'")

# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(reduce_sum)
add_subdirectory(elementwise)