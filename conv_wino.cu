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
  if(row >= 0 && row < N && col >=0 && col < K){
	  c[row * K + col] = 0;
	  for (int k = 0; k < M; k++) {
	    // Accumulate results for a single element
	    c[row * K + col] += a[row * M + k] * b[k * K + col];
	  }
  }
}

__global__ void filter_transform(const int *filters, int *resh_filt, int k, int C, int K){
	//Each thread is responsible for one column of output
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col >=0 && col < K){
		for (int i = 0; i < k*k*C; i++)
			resh_filt[i*K + col] = filters[i + col*k*k*C];	
	}
}

__global__ void feature_transform(const int* features, int *shards, int H, int W, int C, int m = 4){
	uint tsize = m+3-1;
	uint out_rows = (H - 3 + 1) / m;
	uint out_cols = (W - 3 + 1) / m;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (row >= 0 && row < out_rows && col >= 0 && col < out_cols){
		for(int ch = 0; ch < C; ch++){
			for (int u = 0; u < tsize; u++){
				for (int v = 0; v < tsize; v++){
					shards[u*k+v + k*k*C*col + k*k*C*out_cols*row + ch*k*k] = features[ch*H*W + (row*m+u)*W + col*m+v];	
				}
			}
		}
	}
}

void rand_mat(int *a, int size){
	for (int i = 0; i < size; i++)
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

int main(){
	srand(10); //asserting fixed seed for reproducability
	std::string const fname = {"trace_conv_gemm.csv"};
	int dev = 0;
	//Instantiate and start nvml tracing thread
	NVMLMonThread logger(dev, fname);

	int k = 3, C = 8192, K = 16384;
	int H = 56, W = 56;
	int feat_tr_H = (W-k+1)*(H-k+1);
	int feat_tr_W = k*k*C;
	int *kern;
	int *feat;
	int *kern_tr;
	int *feat_tr;
	int *mat_res;

	gpuErrchk(cudaMallocManaged(&kern, k*k*C*K*sizeof(int)));
	gpuErrchk(cudaMallocManaged(&feat, H*W*C*sizeof(int)));
	gpuErrchk(cudaMallocManaged(&kern_tr, k*k*C*K*sizeof(int)));
	gpuErrchk(cudaMallocManaged(&feat_tr, feat_tr_H*feat_tr_W*sizeof(int)));
	gpuErrchk(cudaMallocManaged(&mat_res, feat_tr_H*K*sizeof(int)));

	rand_mat(kern, k*k*C*K);
	rand_mat(feat, H*W*C);

	std::thread threadStart(&NVMLMonThread::log, &logger);

	int THREADS = 32;
	int BLOCKS = (K + THREADS - 1)/THREADS;
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
	int FTTHREADS = 32;
	dim3 threads(FTTHREADS, FTTHREADS);
	int CBLOCKS = (W-k+1 + FTTHREADS - 1) / FTTHREADS;
	int RBLOCKS = (H-k+1 + FTTHREADS - 1) / FTTHREADS;
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
	int THREADS_MUL = 32;
	int BLOCKS_R = (feat_tr_H + THREADS_MUL - 1)/THREADS_MUL;
	int BLOCKS_C = (K + THREADS_MUL - 1)/THREADS_MUL;
	dim3 threads_mul(THREADS_MUL, THREADS_MUL);
	dim3 blocks_mul(BLOCKS_C, BLOCKS_R);
	matrixMul<<<blocks_mul, threads_mul>>>(feat_tr, kern_tr, mat_res, feat_tr_H, feat_tr_W, K);
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
