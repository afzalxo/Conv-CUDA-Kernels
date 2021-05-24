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

//Printing takes quite a bit of time. Discount time logging when debugging
#define DEBUG 0

//Error check adopted from https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void Matmul(const int *a, const int *b, int *c, ulong N, ulong M, ulong K) {
  ulong row = blockIdx.y * blockDim.y + threadIdx.y;
  ulong col = blockIdx.x * blockDim.x + threadIdx.x;
  //printf("Kernel Called...");
  if(row < N && col < K){
	  c[row * K + col] = 0;
	  for (ulong k = 0; k < M; k++) {
	  	c[row * K + col] += a[row * M + k] * b[k * K + col];
	  }
  }
}

//Adopted from https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/matrixMul/tiled/mmul.cu
#define SHMEM_SIZE 1024
__global__ void tiledMatmul(const int *a, const int *b, int *c, uint M, uint K, uint N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ int s_a[SHMEM_SIZE];
  __shared__ int s_b[SHMEM_SIZE];

  int tmp = 0;

  if(row < M && col < N){
	  for (int i = 0; i < K; i += blockDim.x) {
	    s_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * K + i + threadIdx.x];
	    s_b[threadIdx.y * blockDim.x + threadIdx.x] = b[i * N + threadIdx.y * N + col];

	    __syncthreads();

	    for (int j = 0; j < blockDim.x; j++) {
		tmp += s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
	    }

	    __syncthreads();
	  }
  }

  c[row * N + col] = tmp;
}

__global__ void filter_transform(const int *filters, int *resh_filt, ulong k, ulong C, ulong K){
	//Each thread is responsible for one column of output
	ulong col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < K){
		for (ulong i = 0; i < k*k*C; i++)
			resh_filt[i*K + col] = filters[i + col*k*k*C];	
	}
}

__global__ void feature_transform(const int* features, int *shards, ulong H, ulong W, ulong C, ulong k){
	ulong out_rows = H - k + 1;
	ulong out_cols = W - k + 1;
	ulong col = blockIdx.x * blockDim.x + threadIdx.x;
	ulong row = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < out_rows && col < out_cols){
		for(ulong ch = 0; ch < C; ch++){
			for (ulong u = 0; u < k; u++){
				for (ulong v = 0; v < k; v++){
					shards[u*k+v + k*k*C*col + k*k*C*out_cols*row + ch*k*k] = features[ch*H*W + (row + u)*W + col+v];	
				}
			}
		}
	}
}

void rand_mat(int *a, uint size){
	for (uint i = 0; i < size; i++)
		a[i] = rand() % 100;
}

//Unused
void print_mat(int *a, int rows, int cols){
	printf("\n");
	for (int r = 0; r < rows; r++){
		for (int c = 0; c < cols; c++)
			printf("%d ", a[c+r*cols]);
		printf("\n");
	}
	printf("\n");
}

void print_3d_tensor(int *a, int rows, int cols, int channels){
	printf("\n");
	for (int ch = 0; ch < channels; ch++){
		for (int r = 0; r < rows; r++){
			for (int c = 0; c < cols; c++)
				printf("%d ", a[c + r*cols + ch*rows*cols]);
			printf("\n");
		}
		printf("\n-----------\n");
	}
	printf("\n");
}

void print_4d_tensor(int *a, int rows, int cols, int channels, int number){
	printf("\n");
	for (int num = 0; num < number; num++){
		for (int ch = 0; ch < channels; ch++){
			for (int r = 0; r < rows; r++){
				for (int c = 0; c < cols; c++)
					printf("%d ", a[c + r*cols + ch*rows*cols + num*channels*rows*cols]);
				printf("\n");
			}
			printf("\n-----------\n");
		}
		printf("\n+++++++++++\n");
	}
	printf("\n");
}

#define TPB 32

int main(){
	srand(10); //asserting fixed seed for reproducability
	std::string const fname = {"conv_gemm_shmem_32.csv"};
	int dev = 0;
	//Instantiate and start nvml tracing thread
	NVMLMonThread logger(dev, fname);

	ulong k = 3, C = 256, K = 131072;
	ulong H = 224, W = 224;
	ulong feat_tr_H = (W-k+1)*(H-k+1);
	ulong feat_tr_W = k*k*C;
	int *kern;
	int *feat;
	int *kern_tr;
	int *feat_tr;
	int *mat_res;

	std::thread threadStart(&NVMLMonThread::log, &logger);
	logger.caller_state = 0;

	gpuErrchk(cudaMallocManaged(&kern, sizeof(int)*k*k*C*K));
	gpuErrchk(cudaMallocManaged(&feat, sizeof(int)*H*W*C));
	gpuErrchk(cudaMallocManaged(&kern_tr, sizeof(int)*k*k*C*K));
	gpuErrchk(cudaMallocManaged(&feat_tr, sizeof(int)*feat_tr_H*feat_tr_W));
	gpuErrchk(cudaMallocManaged(&mat_res, sizeof(int)*feat_tr_H*K));

	rand_mat(kern, k*k*C*K);
	rand_mat(feat, H*W*C);

	int THREADS = TPB;
	ulong BLOCKS = (K + THREADS - 1)/THREADS;
	logger.caller_state = 1; //Calling filter transform kernel state
	filter_transform<<<BLOCKS, THREADS>>>(kern, kern_tr, k, C, K);
	gpuErrchk(cudaDeviceSynchronize());
	logger.caller_state = 2; //Calling FM transform kernel exec state
#if DEBUG
	printf("Printing origin filters\n");
	print_4d_tensor(kern, k, k, C, K);
	printf("\nPrinting reshaped filters\n");
	print_3d_tensor(kern_tr, k*k*C, K, 1);
#endif
	//int THREADS_C = W-k+1;
	//int THREADS_R = H-k+1;
	//dim3 threads(THREADS_R, THREADS_C);
	int FTTHREADS = TPB;
	dim3 threads(FTTHREADS, FTTHREADS);
	ulong CBLOCKS = (W-k+1 + FTTHREADS - 1) / FTTHREADS;
	ulong RBLOCKS = (H-k+1 + FTTHREADS - 1) / FTTHREADS;
	dim3 blocks(CBLOCKS, RBLOCKS);
	feature_transform<<<blocks, threads>>>(feat, feat_tr, H, W, C, k);
	gpuErrchk(cudaDeviceSynchronize());
	logger.caller_state = 3; //Calling matmul kernel state
#if DEBUG
	printf("\nPrinting original FM\n");
	print_3d_tensor(feat, H, W, C);
	printf("\nPrinting shards\n");
	print_3d_tensor(feat_tr, feat_tr_H, feat_tr_W, 1);
#endif
	int THREADS_MUL = TPB;
	ulong BLOCKS_R = (feat_tr_H + THREADS_MUL - 1)/THREADS_MUL;
	ulong BLOCKS_C = (K + THREADS_MUL - 1)/THREADS_MUL;
	dim3 threads_mul(THREADS_MUL, THREADS_MUL);
	dim3 blocks_mul(BLOCKS_C, BLOCKS_R);
	Matmul<<<blocks_mul, threads_mul>>>(feat_tr, kern_tr, mat_res, feat_tr_H, feat_tr_W, K);
	//tiledMatmul<<<blocks_mul, threads_mul>>>(feat_tr, kern_tr, mat_res, feat_tr_H, feat_tr_W, K);
	gpuErrchk(cudaDeviceSynchronize());
	logger.caller_state = 4; //Finished exec state.
	std::thread threadKill(&NVMLMonThread::killThread, &logger);
	threadStart.join();
	threadKill.join();
#if DEBUG
	printf("\nPrinting results\n");
	print_3d_tensor(mat_res, feat_tr_H, K, 1);
#endif
	cudaFree(kern);
	cudaFree(feat);
	cudaFree(feat_tr);
	cudaFree(kern_tr);
	cudaFree(mat_res);
	
	printf("\n Finished... \n");
	return 0;
}

/*
uint out_rows = uint((H + 2*padding - k) / stride) + 1;
uint out_cols = uint((W + 2*padding - k) / stride) + 1;
uint temp0 = 0, temp1 = 0;
float *reshaped_filters = new float[k*k*C*K];
float *shards = new float[out_rows*out_cols*k*k*C];
//Reshaping filters from [k,k,C,K] to [k*k*C, K]
for(uint c = 0; c < K; c++){
	for(uint r = 0; r < k*k*C; r++){
		reshaped_filters[r*K+c] = kernel[r+c*k*k*C];
	}
}

//Reshaping activations and collecting shards: Shards shape [out_rows*out_cols, k*k*C] when using SGEMM, otherwise [out_rows, out_cols, k*k*C]
for(uint r = 0; r < out_rows; r++){
	for(uint c = 0; c < out_cols; c++){
		for(uint ch = 0; ch < C; ch++){
			for(uint u = 0; u < k; u++){
				for(uint v = 0; v < k; v++){
					temp0 = r + u;
					temp1 = c + v;
					shards[u*k+v + k*k*C*c + k*k*C*out_cols*r + ch*k*k] = data[ch*H*W+(temp0)*W+(temp1)];
				}
			}
		}	
	}
}
*/
