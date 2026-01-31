#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define THREAD_PER_BLOCK 256

__global__ void reduce_sum_naive(float *d_input, float *d_output) {
  volatile __shared__ float shared_mem[THREAD_PER_BLOCK];
  float *input_begin = d_input + blockIdx.x * blockDim.x * 2;
  int tid = threadIdx.x;
  shared_mem[tid] = input_begin[tid] + input_begin[tid + blockDim.x];
  __syncthreads();
#pragma unroll
  for (int i = blockDim.x / 2; i > 32; i /= 2) {
    if (tid < i) {
      shared_mem[tid] += shared_mem[tid + i];
    }
    __syncthreads();
  }
  if (tid < 32) {
    shared_mem[tid] += shared_mem[tid + 32];
    shared_mem[tid] += shared_mem[tid + 16];
    shared_mem[tid] += shared_mem[tid + 8];
    shared_mem[tid] += shared_mem[tid + 4];
    shared_mem[tid] += shared_mem[tid + 2];
    shared_mem[tid] += shared_mem[tid + 1];
  }

  if (tid == 0) {
    d_output[blockIdx.x] = shared_mem[0];
  }
}

bool check(float *res, float *out, const int n) {
  for (int i = 0; i < n; ++i) {
    if (abs(res[i] - out[i]) > 0.001) {
      return false;
    }
  }
  return true;
}

int main() {
  const int N = 3 * 1024 * 1024;
  float *a = new float[N];
  float *d_a;
  cudaMalloc((void **)&d_a, N * sizeof(float));
  const int block_num = N / THREAD_PER_BLOCK / 2;

  float *out = new float[block_num];
  float *d_out;
  cudaMalloc((void **)&d_out, block_num * sizeof(float));

  float *res = new float[block_num];

  for (int i = 0; i < N; i++) {
    a[i] = 2.0 * (float)drand48() - 1.0;
  }

  for (int i = 0; i < block_num; i++) {
    float cur = 0;
    for (int j = 0; j < THREAD_PER_BLOCK * 2; j++) {
      cur += a[i * THREAD_PER_BLOCK * 2 + j];
    }
    res[i] = cur;
  }

  cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
  dim3 grid(block_num, 1);
  dim3 block(THREAD_PER_BLOCK, 1);

  reduce_sum_naive<<<grid, block>>>(d_a, d_out);

  cudaMemcpy(out, d_out, block_num * sizeof(float), cudaMemcpyDeviceToHost);

  // check
  if (check(res, out, block_num)) {
    std::cout << "success" << std::endl;
  } else {
    std::cout << "fail" << std::endl;
  }
  for (int i = 0; i < 16; i++) {
    std::cout << out[i] << " ";
    std::cout << res[i] << std::endl;
  }
  cudaFree(d_a);
  cudaFree(d_out);
  delete[] a;
  delete[] out;
  delete[] res;
  return 0;
}