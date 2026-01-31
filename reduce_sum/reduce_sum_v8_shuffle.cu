#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define THREAD_PER_BLOCK 256
#define WARP_SIZE 32

template <unsigned int block_size, unsigned int NUM_PER_BLOCK,
          unsigned int NUM_PER_THREAD>
__global__ void reduce_sum_naive(float *d_input, float *d_output) {
  float *input_begin = d_input + blockIdx.x * NUM_PER_BLOCK;
  int tid = threadIdx.x;
  float sum = 0.f;
  for (int i = 0; i < NUM_PER_THREAD; i++) {
    sum += input_begin[i * block_size + tid];
  }

  sum += __shfl_down_sync(0xffffffff, sum, 16);
  sum += __shfl_down_sync(0xffffffff, sum, 8);
  sum += __shfl_down_sync(0xffffffff, sum, 4);
  sum += __shfl_down_sync(0xffffffff, sum, 2);
  sum += __shfl_down_sync(0xffffffff, sum, 1);

  __shared__ float warpLevelSums[32];
  const int laneId = tid % WARP_SIZE;
  const int warpId = tid / WARP_SIZE;
  if (laneId == 0) {
    warpLevelSums[warpId] = sum;
  }

  __syncthreads();
  if (warpId == 0) {
    sum = laneId < blockDim.x / WARP_SIZE ? warpLevelSums[laneId] : 0.f;
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
  }
  if (tid == 0) {
    d_output[blockIdx.x] = sum;
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
  const int block_num = 1024; // 有多少个block

  float *out = new float[block_num];
  float *d_out;
  cudaMalloc((void **)&d_out, block_num * sizeof(float));

  float *res = new float[block_num];

  for (int i = 0; i < N; i++) {
    a[i] = 2.0 * (float)drand48() - 1.0;
  }
  const int num_per_block = N / block_num;
  const int num_per_thread = num_per_block / THREAD_PER_BLOCK;
  for (int i = 0; i < block_num; i++) {
    float cur = 0;
    for (int j = 0; j < num_per_block; j++) {
      cur += a[i * num_per_block + j];
    }
    res[i] = cur;
  }

  cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
  dim3 grid(block_num, 1);
  dim3 block(THREAD_PER_BLOCK, 1);

  reduce_sum_naive<THREAD_PER_BLOCK, num_per_block, num_per_thread>
      <<<grid, block>>>(d_a, d_out);

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