#include <cstdlib>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define WARP_SIZE 32
#define BLOCK_SIZE 256

#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])

__global__ void relu_f16_kernel(half *input, half *output, int n) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  if (idx < n) {
    half2 reg_x = HALF2(input[idx]);
    half2 reg_y;
    reg_y.x = __hmax(__float2half(0.0f), reg_x.x);
    reg_y.y = __hmax(__float2half(0.0f), reg_x.y);
    HALF2(output[idx]) = reg_y; 
  }
}

int main() {
  const int N = 1024 * 1024 * 16;
  std::vector<half> h_x(N);
  for (int i = 0; i < N; i++) {
    h_x[i] = __float2half(2.0 * (float)drand48() - 1.0);
  }

  half *d_x, *d_y;
  cudaMalloc((void **)&d_x, N * sizeof(half));
  cudaMalloc((void **)&d_y, N * sizeof(half));
  cudaMemcpy(d_x, h_x.data(), N * sizeof(half), cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  relu_f16_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE / 2>>>(d_x, d_y,
                                                                     N);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  std::vector<half> h_y(N);
  cudaMemcpy(h_y.data(), d_y, N * sizeof(half), cudaMemcpyDeviceToHost);
  for (int i = 0; i < N; i++) {
    if (__half2float(h_y[i]) != fmax(0.0f, __half2float(h_x[i]))) {
      std::cout << "Error at index " << i << ": " << __half2float(h_y[i])
                << " != " << fmax(0.0f, h_x[i]) << std::endl;
      exit(1);
    }
  }
  std::cout << "Success!" << std::endl;
  std::cout << "Time: " << milliseconds << " ms" << std::endl;

  cudaFree(d_x);
  cudaFree(d_y);
  return 0;
}