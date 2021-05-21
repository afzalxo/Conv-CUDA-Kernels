#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include <cuda.h>
#include "nvml_monitor.h"

using std::cout;
using std::generate;
using std::vector;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void matrixMul(const int *a, const int *b, int *c, int N, int M, int K) {
  // Compute each thread's global row and column index
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  //printf("Kernel Called...");

  // Iterate over row, and down column
  //printf("%d, %d\n", row, col);
  if(row < N && col < K){
	  c[row * K + col] = 0;
	  for (int k = 0; k < M; k++) {
	    // Accumulate results for a single element
	    c[row * K + col] += a[row * M + k] * b[k * K + col];
	  }
  }
}

// Check result on the CPU
void verify_result(vector<int> &a, vector<int> &b, vector<int> &c, int N, int M, int K) {
  // For every row...
  for (int i = 0; i < N; i++) {
    // For every column...
    for (int j = 0; j < K; j++) {
      // For every element in the row-column pair
      int tmp = 0;
      for (int k = 0; k < M; k++) {
        // Accumulate the partial results
        tmp += a[i * M + k] * b[k * K + j];
      }

      // Check against the CPU result
      //printf("%d, %d, %d, %d\n", tmp, c[i*K+j], i, j);
      assert(tmp == c[i * K + j]);
    }
  }
}

int main() {
  std::string const fname = {"trace_new.csv"};
  int dev = 0;
  NVMLMonThread logger(dev, fname);
  int N = 1 << 14;
  int M = 1 << 14;
  int K = 1 << 14;

  //Start Monitoring Thraed
  std::thread threadStart(&NVMLMonThread::log, &logger);

  size_t size_a = N * M * sizeof(int);
  size_t size_b = M * K * sizeof(int);
  size_t size_c = N * K * sizeof(int);

  // Host vectors
  vector<int> h_a(N * M);
  vector<int> h_b(M * K);
  vector<int> h_c(N * K);

  // Initialize matrices
  generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
  generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });


  // Allocate device memory
  logger.caller_state = 1;
  int *d_a, *d_b, *d_c;
  gpuErrchk(cudaMallocManaged(&d_a, size_a));
  gpuErrchk(cudaMallocManaged(&d_b, size_b));
  gpuErrchk(cudaMallocManaged(&d_c, size_c));

  // Copy data to the device
  gpuErrchk(cudaMemcpy(d_a, h_a.data(), size_a, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_b, h_b.data(), size_b, cudaMemcpyHostToDevice));

  // Threads per block in one dimension. Using equal number of threads in both dimensions of block.
  int THREADS = 32;

  // Number of blocks in each dimension
  int BLOCKS_X = (K+THREADS-1) / THREADS;
  int BLOCKS_Y = (N+THREADS-1) / THREADS;

  // Use dim3 structs for block  and grid dimensions
  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS_X, BLOCKS_Y);

  // Launch kernel
  printf("Calling Kernel...\n");
  logger.caller_state = 2;
  matrixMul<<<blocks, threads>>>(d_a, d_b, d_c, N, M, K);
  gpuErrchk(cudaDeviceSynchronize());

  logger.caller_state = 3;
  // Copy back to the host
  gpuErrchk(cudaMemcpy(h_c.data(), d_c, size_c, cudaMemcpyDeviceToHost));

  cudaDeviceSynchronize();
  // Check result
  //printf("Verifying Result...\n");
  //verify_result(h_a, h_b, h_c, N, M, K);

  cout << "COMPLETED SUCCESSFULLY\n";

  // Free memory on device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  
  logger.caller_state = 4;
  std::thread threadKill(&NVMLMonThread::killThread, &logger);
  //logger.loop = false;
  threadStart.join();
  threadKill.join();

  return 0;
}

