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
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  half2 reg_x_0 = HALF2(input[idx]);
  half2 reg_x_1 = HALF2(input[idx + 2]);
  half2 reg_x_2 = HALF2(input[idx + 4]);
  half2 reg_x_3 = HALF2(input[idx + 6]);
  half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;
  reg_y_0.x = __hmax(__float2half(0.0f), reg_x_0.x);
  reg_y_0.y = __hmax(__float2half(0.0f), reg_x_0.y);
  reg_y_1.x = __hmax(__float2half(0.0f), reg_x_1.x);
  reg_y_1.y = __hmax(__float2half(0.0f), reg_x_1.y);
  reg_y_2.x = __hmax(__float2half(0.0f), reg_x_2.x);
  reg_y_2.y = __hmax(__float2half(0.0f), reg_x_2.y);
  reg_y_3.x = __hmax(__float2half(0.0f), reg_x_3.x);
  reg_y_3.y = __hmax(__float2half(0.0f), reg_x_3.y);

  if (idx < n) {
    HALF2(output[idx]) = reg_y_0;
  }
  if (idx + 2 < n) {
    HALF2(output[idx + 2]) = reg_y_1;
  }
  if (idx + 4 < n) {
    HALF2(output[idx + 4]) = reg_y_2;
  }
  if (idx + 6 < n) {
    HALF2(output[idx + 6]) = reg_y_3;
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

  relu_f16_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE / 8>>>(
      d_x, d_y, N);

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