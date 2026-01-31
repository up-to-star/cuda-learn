add_executable(elementwise_addf16_v0_naive elementwise_addf16_v0_naive.cu)

target_link_libraries(elementwise_addf16_v0_naive PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(elementwise_addf16_v0_naive PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()