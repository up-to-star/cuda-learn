#include <cstdlib>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define WARP_SIZE 32
#define BLOCK_SIZE 512

#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

__global__ void elementwise_addf16(__half *a, __half *b, __half *c, int n) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  half pack_a[8], pack_b[8], pack_c[8];
  LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);
  LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]);
#pragma unroll
  for (int i = 0; i < 8; i += 2) {
    HALF2(pack_c[i]) = __hadd2(HALF2(pack_a[i]), HALF2(pack_b[i]));
  }
  if (idx + 7 < n) {
    LDST128BITS(c[idx]) = LDST128BITS(pack_c[0]);
  } else {
    for (int i = 0; idx + i < n; i++) {
      c[idx + i] = __hadd(a[idx + i], b[idx + i]);
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