#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include <cuda.h>
#include "nvml_monitor.h"
//#include "gemm_kernel.h"

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

__global__ void filter_transform(const int *filters, int *resh_filt, int k, int C, int K){
	//Each thread is responsible for one column of output
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col >=0 && col < K){
		for (int i = 0; i < k*k*C; i++)
			resh_filt[i*K + col] = filters[i + col*k*k*C];	
	}
}

__global__ void feature_transform(const int* features, int *shards, int H, int W, int C, int k){
	uint out_rows = H - k + 1;
	uint out_cols = W - k + 1;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (row >= 0 && row < out_rows && col >= 0 && col < out_cols){
		for(int ch = 0; ch < C; ch++){
			for (int u = 0; u < k; u++){
				for (int v = 0; v < k; v++){
					shards[u*k+v + k*k*C*col + k*k*C*out_cols*row + ch*k*k] = features[ch*H*W + (row + u)*W + col+v];	
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
	int k = 3, C = 2, K = 2;
	int H = 3, W = 3;
	int feat_tr_H = (W-k+1)*(H-k+1);
	int feat_tr_W = k*k*C;
	int *kern;
	int *feat;
	int *kern_tr;
	int *feat_tr;
	int *mat_res;
	gpuErrchk(cudaMallocManaged(&kern, k*k*C*K));
	gpuErrchk(cudaMallocManaged(&feat, H*W*C));
	gpuErrchk(cudaMallocManaged(&kern_tr, k*k*C*K));
	gpuErrchk(cudaMallocManaged(&feat_tr, feat_tr_H*feat_tr_W));
	gpuErrchk(cudaMallocManaged(&mat_res, feat_tr_H*feat_tr_W*K));
	rand_mat(kern, k*k*C*K);
	rand_mat(feat, H*W*C);
	int THREADS=9;
	int BLOCKS=1;
	filter_transform<<<BLOCKS, THREADS>>>(kern, kern_tr, k, C, K);
	printf("Printing origin filters\n");
	print_4d_tensor(kern, k, k, C, K);
	//for (int i = 0; i < k*k*C*K; i++){
	//	printf("%d ", kern[i]);
	//}
	gpuErrchk(cudaDeviceSynchronize());
	printf("\nPrinting reshaped filters\n");
	print_3d_tensor(kern_tr, k*k*C, K, 1);
	//for (int i = 0; i < k*k*C*K; i++){
	//	printf("%d ", kern_tr[i]);
	//}
	printf("\n");
	int THREADS_C = W-k+1;
	int THREADS_R = H-k+1;
	dim3 threads(THREADS_C, THREADS_R);
	dim3 blocks(1);
	feature_transform<<<blocks, threads>>>(feat, feat_tr, H, W, C, k);
	printf("\nPrinting original FM\n");
	print_3d_tensor(feat, H, W, C);
	gpuErrchk(cudaDeviceSynchronize());
	printf("\nPrinting shards\n");
	print_3d_tensor(feat_tr, feat_tr_H, feat_tr_W, 1);

	int THREADS_MUL = 32;
	int BLOCKS_R = (feat_tr_H*feat_tr_W + THREADS_MUL - 1)/THREADS_MUL;
	int BLOCKS_C = (K + THREADS_MUL - 1)/THREADS_MUL;
	dim3 threads_mul(THREADS_MUL, THREADS_MUL);
	dim3 blocks_mul(BLOCKS_R, BLOCKS_C);
	matrixMul<<<blocks_mul, threads_mul>>>(feat_tr, kern_tr, mat_res, feat_tr_H, feat_tr_W, K);
	gpuErrchk(cudaDeviceSynchronize());
	printf("\nPrinting results\n");
	print_3d_tensor(mat_res, feat_tr_H, K, 1);
	
	cudaFree(kern);
	cudaFree(feat);
	cudaFree(feat_tr);
	cudaFree(kern_tr);
	cudaFree(mat_res);
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
