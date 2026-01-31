
add_executable(reduce_sum reduce_sum_v0_naive.cu)

target_link_libraries(reduce_sum PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_sum PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(reduce_sum_shared_memory reduce_sum_v1_shared_memory.cu)

target_link_libraries(reduce_sum_shared_memory PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_sum_shared_memory PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(reduce_sum_v2_no_divergence reduce_sum_v2_no_divergence.cu)

target_link_libraries(reduce_sum_v2_no_divergence PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_sum_v2_no_divergence PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(reduce_sum_v3_no_bank_conflict reduce_sum_v3_no_bank_conflict.cu)

target_link_libraries(reduce_sum_v3_no_bank_conflict PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_sum_v3_no_bank_conflict PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(reduce_sum_v4_no_idle_thread reduce_sum_v4_no_idle_thread.cu)

target_link_libraries(reduce_sum_v4_no_idle_thread PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_sum_v4_no_idle_thread PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(reduce_sum_v5_unroll_last_warp reduce_sum_v5_unroll_last_warp.cu)

target_link_libraries(reduce_sum_v5_unroll_last_warp PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_sum_v5_unroll_last_warp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(reduce_sum_v6_complete_unroll reduce_sum_v6_complete_unroll.cu)

target_link_libraries(reduce_sum_v6_complete_unroll PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_sum_v6_complete_unroll PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(reduce_sum_v7_multi_add reduce_sum_v7_multi_add.cu)

target_link_libraries(reduce_sum_v7_multi_add PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_sum_v7_multi_add PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(reduce_sum_v8_shuffle reduce_sum_v8_shuffle.cu)

target_link_libraries(reduce_sum_v8_shuffle PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_sum_v8_shuffle PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()