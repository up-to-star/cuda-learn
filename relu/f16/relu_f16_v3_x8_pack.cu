#include <cstdlib>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define WARP_SIZE 32
#define BLOCK_SIZE 256

#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

__global__ void relu_f16_kernel(half *input, half *output, int n) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  half pack_x[8], pack_y[8];
  LDST128BITS(pack_x[0]) = LDST128BITS(input[idx]);
  const half2 z2 = {__float2half(0.0f), __float2half(0.0f)};
  for (int i = 0; i < 8; i += 2) {
    HALF2(pack_y[i]) = __hmax2(HALF2(pack_x[i]), z2);
  }
  if (idx + 7 < n) {
    LDST128BITS(output[idx]) = LDST128BITS(pack_y[0]);
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