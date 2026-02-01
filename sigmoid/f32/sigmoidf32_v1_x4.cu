#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define MAX_EXP_F32 88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

__global__ void sigmoid_f32_kernel(float *x, float *y, int n) {
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  float4 reg_x = FLOAT4(x[idx]), reg_y;
  reg_x.x = fmin(fmax(reg_x.x, MIN_EXP_F32), MAX_EXP_F32);
  reg_x.y = fmin(fmax(reg_x.y, MIN_EXP_F32), MAX_EXP_F32);
  reg_x.z = fmin(fmax(reg_x.z, MIN_EXP_F32), MAX_EXP_F32);
  reg_x.w = fmin(fmax(reg_x.w, MIN_EXP_F32), MAX_EXP_F32);
  reg_y.x = 1.0f / (1.0f + expf(-reg_x.x));
  reg_y.y = 1.0f / (1.0f + expf(-reg_x.y));
  reg_y.z = 1.0f / (1.0f + expf(-reg_x.z));
  reg_y.w = 1.0f / (1.0f + expf(-reg_x.w));
  if (idx < n) {
    FLOAT4(y[idx]) = reg_y;
  }
}

int main() {
  const int n = 1024 * 1024 * 16;
  size_t bytes = n * sizeof(float);
  std::vector<float> h_x(n), h_y(n);
  for (int i = 0; i < n; i++) {
    h_x[i] = 2.0 * (float)drand48() - 1.0;
  }
  float *d_x, *d_y;
  cudaMalloc(&d_x, bytes);
  cudaMalloc(&d_y, bytes);
  cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  sigmoid_f32_kernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE / 4>>>(d_x,
                                                                        d_y, n);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost);
  bool correct = true;
  for (int i = 0; i < n; i++) {
    float v = h_x[i];
    v = fmin(fmax(v, MIN_EXP_F32), MAX_EXP_F32);
    float expected = 1.0f / (1.0f + expf(-v));
    if (std::abs(h_y[i] - expected) > 1e-3) {
      std::cout << "Mismatch at " << i << ": " << h_y[i] << " vs " << expected
                << std::endl;
      correct = false;
      break;
    }
  }
  if (correct) {
    std::cout << "All results are correct." << std::endl;
    std::cout << "Time: " << milliseconds << " ms" << std::endl;
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_x);
  cudaFree(d_y);
}