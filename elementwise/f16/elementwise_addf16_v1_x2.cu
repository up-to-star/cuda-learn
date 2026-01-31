#include <cstdlib>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define WARP_SIZE 32
#define BLOCK_SIZE 256

#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])

__global__ void elementwise_addf16(__half *a, __half *b, __half *c, int n) {
  int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < n) {
    half2 reg_a = HALF2(a[idx]);
    half2 reg_b = HALF2(b[idx]);
    half2 reg_c;
    reg_c.x = __hadd(reg_a.x, reg_b.x);
    reg_c.y = __hadd(reg_a.y, reg_b.y);
    HALF2(c[idx]) = reg_c;
  }
}

int main() {
  const int N = 1024 * 1024 * 16;
  __half *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, N * sizeof(__half));
  cudaMalloc(&d_b, N * sizeof(__half));
  cudaMalloc(&d_c, N * sizeof(__half));

  std::vector<__half> h_a(N), h_b(N), h_c_gpu(N);

  for (int i = 0; i < N; i++) {
    h_a[i] = __float2half(2.0 * (float)drand48() - 1.0);
    h_b[i] = __float2half(2.0 * (float)drand48() - 1.0);
  }

  cudaMemcpy(d_a, h_a.data(), N * sizeof(__half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), N * sizeof(__half), cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  elementwise_addf16<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE / 2>>>(
      d_a, d_b, d_c, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(h_c_gpu.data(), d_c, N * sizeof(__half), cudaMemcpyDeviceToHost);

  bool correct = true;
  for (int i = 0; i < N; i++) {
    float diff = fabs(__half2float(h_c_gpu[i]) - __half2float(h_a[i] + h_b[i]));
    if (diff > 1e-3) {
      correct = false;
      std::cerr << "Error at index " << i << ": " << diff << std::endl;
      break;
    }
  }

  if (correct) {
    std::cout << "Correct!" << std::endl;
    std::cout << "Time: " << milliseconds << " ms" << std::endl;
  } else {
    std::cout << "Incorrect!" << std::endl;
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}