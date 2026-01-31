add_executable(elementwise_add_v0_naive elementwise_add_v0_naive.cu)

target_link_libraries(elementwise_add_v0_naive PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(elementwise_add_v0_naive PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(elementwise_add_v1_float4 elementwise_add_v1_float4.cu)

target_link_libraries(elementwise_add_v1_float4 PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(elementwise_add_v1_float4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()