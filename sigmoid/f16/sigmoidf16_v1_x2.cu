#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <iostream>
#include <vector>

#define WARP_SIZE 32
#define BLOCK_SIZE 256

#define MAX_EXP_F16 __float2half(11.089866488461016f)
#define MIN_EXP_F16 __float2half(-9.704060527839234f)
#define hexp(value) __float2half(expf(__half2float(value)))
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])

__global__ void sigmoid_f16_kernel(half *x, half *y, int n) {
  int idx = 2 * (threadIdx.x + blockIdx.x * blockDim.x);
  half f = __float2half(1.0f);
  if (idx < n) {
    half2 reg_x = HALF2(x[idx]);
    half2 reg_y;
    reg_x.x = __hmin(__hmax(reg_x.x, MIN_EXP_F16), MAX_EXP_F16);
    reg_x.y = __hmin(__hmax(reg_x.y, MIN_EXP_F16), MAX_EXP_F16);
    reg_y.x = f / (f + hexp(-reg_x.x));
    reg_y.y = f / (f + hexp(-reg_x.y));
    HALF2(y[idx]) = reg_y;
  }
}

int main() {
  const int N = 1024 * 1024 * 16;
  __half *d_x, *d_y;
  cudaMalloc(&d_x, N * sizeof(__half));
  cudaMalloc(&d_y, N * sizeof(__half));

  std::vector<__half> h_x(N), h_y(N);

  for (int i = 0; i < N; i++) {
    h_x[i] = __float2half(2.0 * (float)drand48() - 1.0);
  }

  cudaMemcpy(d_x, h_x.data(), N * sizeof(__half), cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  sigmoid_f16_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE / 2>>>(
      d_x, d_y, N);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  bool correct = true;
  cudaMemcpy(h_y.data(), d_y, N * sizeof(__half), cudaMemcpyDeviceToHost);
  for (int i = 0; i < N; i++) {
    float diff =
        fabs(__half2float(h_y[i]) - 1.0 / (1.0 + expf(-__half2float(h_x[i]))));
    if (diff > 0.001) {
      std::cout << "Error at index " << i << " expected "
                << __half2float(h_y[i]) << " got "
                << __half2float(
                       __float2half(1.0 / (1.0 + expf(-__half2float(h_x[i])))))
                << std::endl;
      correct = false;
      break;
    }
  }
  if (correct) {
    std::cout << "Correct!" << std::endl;
    std::cout << "Elapsed time: " << milliseconds << " ms" << std::endl;
  }

  return 0;
}