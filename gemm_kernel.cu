#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include <cuda.h>

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

  int N = 1 << 10;
  int M = 1 << 10;
  int K = 1 << 10;

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
  int *d_a, *d_b, *d_c;
  gpuErrchk(cudaMallocManaged(&d_a, size_a));
  gpuErrchk(cudaMallocManaged(&d_b, size_b));
  gpuErrchk(cudaMallocManaged(&d_c, size_c));

  // Copy data to the device
  gpuErrchk(cudaMemcpy(d_a, h_a.data(), size_a, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_b, h_b.data(), size_b, cudaMemcpyHostToDevice));

  // Threads per CTA dimension
  int THREADS = 32;

  // Blocks per grid dimension (assumes THREADS divides N evenly)
  int BLOCKS_X = (K+THREADS-1) / THREADS;
  int BLOCKS_Y = (N+THREADS-1) / THREADS;

  // Use dim3 structs for block  and grid dimensions
  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS_X, BLOCKS_Y);

  // Launch kernel
  printf("Calling Kenrel...\n");
  matrixMul<<<blocks, threads>>>(d_a, d_b, d_c, N, M, K);
  gpuErrchk(cudaDeviceSynchronize());

  // Copy back to the host
  gpuErrchk(cudaMemcpy(h_c.data(), d_c, size_c, cudaMemcpyDeviceToHost));

  cudaDeviceSynchronize();
  // Check result
  verify_result(h_a, h_b, h_c, N, M, K);

  cout << "COMPLETED SUCCESSFULLY\n";

  // Free memory on device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}

