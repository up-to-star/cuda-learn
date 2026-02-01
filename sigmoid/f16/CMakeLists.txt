add_executable(sigmoidf16_v0_naive sigmoidf16_v0_naive.cu)

target_link_libraries(sigmoidf16_v0_naive PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(sigmoidf16_v0_naive PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(sigmoidf16_v1_x2 sigmoidf16_v1_x2.cu)

target_link_libraries(sigmoidf16_v1_x2 PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(sigmoidf16_v1_x2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(sigmoidf16_v2_x8 sigmoidf16_v2_x8.cu)

target_link_libraries(sigmoidf16_v2_x8 PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(sigmoidf16_v2_x8 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(sigmoidf16_v3_x8_pack sigmoidf16_v3_x8_pack.cu)

target_link_libraries(sigmoidf16_v3_x8_pack PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(sigmoidf16_v3_x8_pack PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()