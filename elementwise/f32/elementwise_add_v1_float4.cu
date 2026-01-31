#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

__global__ void elementwise_add(float *A, float *B, float *C, const int N) {
  int idx = 4 * (threadIdx.x + blockIdx.x * blockDim.x);
  if (idx < N) {
    float4 reg_a = FLOAT4(A[idx]);
    float4 reg_b = FLOAT4(B[idx]);
    float4 reg_c;
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    reg_c.z = reg_a.z + reg_b.z;
    reg_c.w = reg_a.w + reg_b.w;
    FLOAT4(C[idx]) = reg_c;
  }
}

bool check(const float *res, const float *out, const int N) {
  for (int i = 0; i < N; i++) {
    if (abs(res[i] - out[i]) > 0.001) {
      return false;
    }
  }
  return true;
}

int main() {
  const int N = 1024 * 1024 * 3;
  const int bytes = N * sizeof(float);

  float *h_A = new float[N];
  float *h_B = new float[N];
  float *h_C = new float[N];

  for (int i = 0; i < N; i++) {
    h_A[i] = 2.0 * (float)drand48() - 1.0;
    h_B[i] = 2.0 * (float)drand48() - 1.0;
  }

  float *d_A, *d_B, *d_C;
  cudaMalloc((void **)&d_A, bytes);
  cudaMalloc((void **)&d_B, bytes);
  cudaMalloc((void **)&d_C, bytes);

  cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  elementwise_add<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE / 4>>>(
      d_A, d_B, d_C, N);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

  // check result
  float *d_C_ref = new float[N];
  for (int i = 0; i < N; i++) {
    d_C_ref[i] = h_A[i] + h_B[i];
  }

  if (check(h_C, d_C_ref, N)) {
    std::cout << "Success" << std::endl;
    std::cout << "Time: " << milliseconds << " ms" << std::endl;
  } else {
    std::cout << "Fail" << std::endl;
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  delete[] d_C_ref;
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  return 0;
}