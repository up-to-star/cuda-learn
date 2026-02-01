#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define WARP_SIZE 32
#define BLOCK_SIZE 256

__global__ void relu_f32_kernel(float *input, float *output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = fmax(0.0f, input[idx]);
  }
}

int main() {
  const int N = 1024 * 1024 * 16;
  std::vector<float> h_x(N);
  for (int i = 0; i < N; i++) {
    h_x[i] = 2.0 * (float)drand48() - 1.0;
  }

  float *d_x, *d_y;
  cudaMalloc((void **)&d_x, N * sizeof(float));
  cudaMalloc((void **)&d_y, N * sizeof(float));
  cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  relu_f32_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_x, d_y,
                                                                     N);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  std::vector<float> h_y(N);
  cudaMemcpy(h_y.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < N; i++) {
    if (h_y[i] != fmax(0.0f, h_x[i])) {
      std::cout << "Error at index " << i << ": " << h_y[i]
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