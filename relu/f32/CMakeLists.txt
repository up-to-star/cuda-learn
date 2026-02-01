add_executable(relu_f32_v0_naive relu_f32_v0_naive.cu)

target_link_libraries(relu_f32_v0_naive PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(relu_f32_v0_naive PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(relu_f32_v1_x4 relu_f32_v1_x4.cu)

target_link_libraries(relu_f32_v1_x4 PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(relu_f32_v1_x4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()