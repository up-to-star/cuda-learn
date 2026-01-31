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
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < n) {
    half2 reg_a_0 = HALF2(a[idx]);
    half2 reg_a_1 = HALF2(a[idx + 2]);
    half2 reg_a_2 = HALF2(a[idx + 4]);
    half2 reg_a_3 = HALF2(a[idx + 6]);
    half2 reg_b_0 = HALF2(b[idx]);
    half2 reg_b_1 = HALF2(b[idx + 2]);
    half2 reg_b_2 = HALF2(b[idx + 4]);
    half2 reg_b_3 = HALF2(b[idx + 6]);
    half2 reg_c_0, reg_c_1, reg_c_2, reg_c_3;
    reg_c_0.x = __hadd(reg_a_0.x, reg_b_0.x);
    reg_c_0.y = __hadd(reg_a_0.y, reg_b_0.y);
    reg_c_1.x = __hadd(reg_a_1.x, reg_b_1.x);
    reg_c_1.y = __hadd(reg_a_1.y, reg_b_1.y);
    reg_c_2.x = __hadd(reg_a_2.x, reg_b_2.x);
    reg_c_2.y = __hadd(reg_a_2.y, reg_b_2.y);
    reg_c_3.x = __hadd(reg_a_3.x, reg_b_3.x);
    reg_c_3.y = __hadd(reg_a_3.y, reg_b_3.y);
    if (idx + 0 < n) {
      HALF2(c[idx]) = reg_c_0;
    }
    if (idx + 2 < n) {
      HALF2(c[idx + 2]) = reg_c_1;
    }
    if (idx + 4 < n) {
      HALF2(c[idx + 4]) = reg_c_2;
    }
    if (idx + 6 < n) {
      HALF2(c[idx + 6]) = reg_c_3;
    }
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

  elementwise_addf16<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE / 8>>>(
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