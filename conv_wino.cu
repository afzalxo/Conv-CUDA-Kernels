#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include <cuda.h>
#include <math.h>
#include "nvml_monitor.h"

using std::cout;
using std::generate;
using std::vector;

//template <typename T>
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


__host__ __device__ void tile_ewm(const int *fm, const float *km, float *res, int m = 4){ 
	int tileSize = m+3-1;
	for(int j = 0; j < tileSize; j++)
		res[j] = fm[j] * km[j];
}

__global__ void ewm(const int *fm, const float *km, float *om, ulong H, ulong K, uint C, uint m = 4){
	ulong tile = blockIdx.y * blockDim.y + threadIdx.y;
	ulong row = blockIdx.x * blockDim.x + threadIdx.x;
	ulong kern = blockIdx.z * blockDim.z + threadIdx.z;
	ulong tiles = (H - 3 + 1)/m;
	ulong ntiles = tiles*tiles;
	ulong tileSize = m + 3 -1;
	ulong tilesq = tileSize*tileSize;
	if (row < tileSize && tile < ntiles && kern < K){
		///printf("%d,%d,%d\n",row, col, kern);
		for(uint ch = 0; ch < C; ch ++)
			tile_ewm(&fm[tilesq*ntiles*ch + tile*tilesq + row*tileSize], &km[tilesq*C*kern + ch*tilesq + row*tileSize], &om[tilesq*ntiles*C*kern + tilesq*ntiles*ch + tile*tilesq + row*tileSize], m);
	}
}

__host__ __device__ void tile_inv_transform(const float *fm, float *res, int m = 4){
	int temp0[4*6] = {0};
	int A_tr[4*6] = {1, 1, 1, 1, 1, 0, 0, 1, -1, 2, -2, 0, 0, 1, 1, 4, 4, 0, 0, 1, -1, 8, -8, 1};	
	int A[6*4] = {1, 0, 0, 0, 1, 1, 1, 1, 1, -1, 1, -1, 1, 2, 4, 8, 1, -2, 4, -8, 0, 0, 0, 1};
	//Performing A_tr \times M
	for(int i = 0; i < 4; i++){
		for(int j = 0; j < 6; j++){
			for (int k = 0; k < 6; k++){
				temp0[i*6 + k] += A_tr[i*6 + j] * fm[j*6 + k];
			}
		}
	}
	//Generating A_tr \times M \times A
	for(int i = 0; i < 4; i++){
		for(int j = 0; j < 6; j++){
			for (int k = 0; k < 4; k++){
				res[i*4 + k] += temp0[i*6 + j] * A[j*4 + k];
			}
		}
	}
}

__global__ void inverse_transform(const float *M, float *res, int H, int K, int m = 4){
	ulong row = blockIdx.y * blockDim.y + threadIdx.y;
	ulong col = blockIdx.x * blockDim.x + threadIdx.x;
	ulong kern = blockIdx.z * blockDim.z + threadIdx.z;
	ulong tiles = (H - 3 + 1)/m;
	ulong tileSize = m + 3 -1;
	if (row < tiles && col < tiles && kern < K){
		tile_inv_transform(&M[(row*tiles+col)*tileSize*tileSize + tileSize*tileSize*tiles*tiles*kern], &res[(row*tiles+col)*m*m + m*m*tiles*tiles*kern], m);
	}		
}

__host__ __device__ void transform_filter_tile(const int *temp, float *res, int m = 4){
	float G_tr[18] = {1.0/4.0, -1.0/6.0, -1.0/6.0, 1.0/24.0, 1.0/24.0, 0.0, 0.0, -1.0/6.0, 1.0/6.0, 1.0/12.0, -1.0/12.0, 0.0, 0.0, -1.0/6.0, -1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0};
	float G[18] = {1.0/4.0, 0.0, 0.0, -0.16666667, -0.16666667, -0.16666667, -0.16666667,  0.16666667, -0.16666667, 0.04166667,  0.08333333,  0.16666667, 0.04166667, -0.08333333,  0.16666667, 0.0, 0.0,1.0 };
	float temp0[18] = {0.0};
	//Performing Gg
	for (int i = 0; i < 6; i++){
		for (int j = 0; j< 3; j++){
			for (int k = 0; k < 3; k++){
				temp0[i*3 + k] += G[i*3 + j] * temp[j*3+k];
			}
		}
	}
	//Generating GgG_tr
	for (int i = 0; i < 6; i++){
		for (int j = 0; j< 3; j++){
			for (int k = 0; k < 6; k++){
				res[i*6 + k] += temp0[i*3 + j] * G_tr[j*6+k];
			}
		}
	}
}

__global__ void filter_transform(const int *filters, float *resh_filt, int k, int C, int K){
	//Each thread is responsible for one filter
	ulong col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < K){
		for(int ch = 0; ch < C; ch++)
			transform_filter_tile(&filters[9*ch + 9*C*col], &resh_filt[36*ch + 36*C*col], 4);
	}
}

//Constant matrix muls=> Optimized by compiler into shifts and adds
__host__ __device__ void transform_feature_tile(int *temp, int *res, int m= 4){
	int B_tr[36] = {4, 0, -5, 0, 1, 0, 0, -4, -4, 1, 1, 0, 0, 4, -4, -1, 1, 0, 0, -2, -1, 2, 1, 0, 0, 2, -1, -2, 1, 0, 0, 4, 0, -5, 0, 1};
	int B[36] = {4,  0,  0,  0,  0,  0,  0, -4,  4, -2,  2,  4, -5, -4, -4, -1, -1, 0,  0,  1, -1,  2, -2, -5, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1};
	int temp0[36] = {0};
	for (int i = 0; i < 6; i++){
		for(int j = 0; j < 6; j++){
			for ( int k = 0; k < 6; k++){
				temp0[i*6 + k] += B_tr[i*6 + j] * temp[j*6 + k];
			}		
		}
	}
	for (int i = 0; i < 6; i++){
		for(int j = 0; j < 6; j++){
			for ( int k = 0; k < 6; k++){
				res[i*6 + k] += temp0[i*6 + j] * B[j*6 + k];
			}		
		}
	}
}

__global__ void feature_transform(const int* features, int *shards, int H, int W, int C, int m = 4){
	ulong tsize = m+3-1;
	ulong out_rows = (H - 3 + 1) / m;
	ulong out_cols = (W - 3 + 1) / m;
	ulong col = blockIdx.x * blockDim.x + threadIdx.x;
	ulong row = blockIdx.y * blockDim.y + threadIdx.y;
	int temp[36] = {0};
	int ft_tile[36] = {0};
	//printf("%d, %d\n", row, col);
	if (row < out_rows && col < out_cols){
		for(int ch = 0; ch < C; ch++){
			for (int u = 0; u < tsize; u++){
				for (int v = 0; v < tsize; v++){
					temp[u*tsize + v] = features[H*W*ch + (row*W + col)*m + u*W + v];
				}
			}
			transform_feature_tile(temp, ft_tile, m);
		 	for (int u =0; u < tsize; u++){
				for (int v = 0; v < tsize; v++){
					shards[ch*tsize*tsize*out_rows*out_cols + (row*out_cols + col) * tsize*tsize + u*tsize + v] = ft_tile[u*tsize + v];
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

void print_3d_tensor_float(float *a, int rows, int cols, int channels){
	printf("\n");
	for (int ch = 0; ch < channels; ch++){
		for (int r = 0; r < rows; r++){
			for (int c = 0; c < cols; c++)
				printf("%f ", a[c + r*cols + ch*rows*cols]);
			printf("\n");
		}
		printf("\n-----------\n");
	}
	printf("\n");
}
void print_3d_tensor_int(int *a, int rows, int cols, int channels){
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

#define TPB 4

int main(){
	srand(10); //asserting fixed seed for reproducability
	std::string const fname = {"trace_conv_wino_4.csv"};
	int dev = 0;
	//Instantiate and start nvml tracing thread
	NVMLMonThread logger(dev, fname);

	ulong k = 3, C = 1, K = 131072;
	ulong H = 224, W = 224, m = 4;
	uint tileSize = m + k - 1;
	ulong feat_tiles_per_ch_horiz = (W - k + 1) / m;
	ulong feat_tiles_per_ch_vert = (H - k + 1) / m;
	ulong feat_tiles_per_ch = feat_tiles_per_ch_horiz * feat_tiles_per_ch_vert;
	int *kern;
	int *feat;
	float *kern_tr;
	int *feat_tr;
	float *ewm_res;
	//float *conv_out;

	gpuErrchk(cudaMallocManaged(&feat, sizeof(int)*H*W*C));
	gpuErrchk(cudaMallocManaged(&feat_tr, sizeof(int)*tileSize*tileSize*feat_tiles_per_ch*C));
	gpuErrchk(cudaMallocManaged(&kern, sizeof(int)*k*k*C*K));
	gpuErrchk(cudaMallocManaged(&kern_tr, sizeof(int)*tileSize*tileSize*C*K));

	rand_mat(kern, k*k*C*K);
	rand_mat(feat, H*W*C);

	std::thread threadStart(&NVMLMonThread::log, &logger);
	
	int THREADS = TPB;
	int BLOCKS = (K + THREADS - 1)/THREADS;
	logger.caller_state = 1; //Calling filter transform kernel state
	filter_transform<<<BLOCKS, THREADS>>>(kern, kern_tr, k, C, K);
	gpuErrchk(cudaDeviceSynchronize()); //Dont need to block exec here since feature_transform is indep of filter transform.
	logger.caller_state = 2; //Calling FM transform kernel exec state
#if DEBUG
	printf("Printing original filters\n");
	print_4d_tensor(kern, k, k, C, K);
	printf("\nPrinting reshaped filters\n");
	print_3d_tensor_float(kern_tr, tileSize, tileSize, K);
#endif
	
	//int THREADS_C = W-k+1;
	//int THREADS_R = H-k+1;
	//dim3 threads(THREADS_R, THREADS_C);

	//float *fil_out;
	//gpuErrchk(cudaMallocManaged(&fil_out, 6*6*sizeof(float)));
	//transform_filter_tile(kern, fil_out, 4);
	//print_4d_tensor(kern, k, k, C, K);
	//print_3d_tensor(fil_out, 6,6,1);

	//int *ft_out;
	//gpuErrchk(cudaMallocManaged(&ft_out, 6*6*sizeof(int)));
	//transform_feature_tile(feat, ft_out, 4);
	//print_3d_tensor(feat, H, W, C);
	//print_3d_tensor(ft_out, 6,6,1);
		
	int FTTHREADS = TPB;//32;
	dim3 threads(FTTHREADS, FTTHREADS);
	int CBLOCKS = ((W-k+1)/m + FTTHREADS - 1) / FTTHREADS;
	int RBLOCKS = ((H-k+1)/m + FTTHREADS - 1) / FTTHREADS;
	dim3 blocks(CBLOCKS, RBLOCKS);
	feature_transform<<<blocks, threads>>>(feat, feat_tr, H, W, C);
	gpuErrchk(cudaDeviceSynchronize());
	logger.caller_state = 3; //Calling ewm kernel state
#if DEBUG
	printf("\nPrinting original FM\n");
	print_3d_tensor_int(feat, H, W, C);
	printf("\nPrinting shards\n");
	print_3d_tensor_int(feat_tr, tileSize, tileSize, feat_tiles_per_ch);
#endif
	cudaFree(feat);
	cudaFree(kern);
	gpuErrchk(cudaMallocManaged(&ewm_res, sizeof(float)*feat_tiles_per_ch*tileSize*tileSize*C*K));
	gpuErrchk(cudaDeviceSynchronize());
	int THREADS_MUL = TPB;//8;
	int BLOCKS_X = (tileSize + THREADS_MUL - 1)/THREADS_MUL;
	int BLOCKS_Y = (feat_tiles_per_ch + THREADS_MUL - 1)/THREADS_MUL;
	int BLOCKS_Z = (K + THREADS_MUL - 1)/THREADS_MUL;
	dim3 threads_mul(THREADS_MUL, THREADS_MUL, THREADS_MUL);
	dim3 blocks_mul(BLOCKS_X, BLOCKS_Y, BLOCKS_Z);
	ewm<<<blocks_mul, threads_mul>>>(feat_tr, kern_tr, ewm_res, H, K, C, m);
	gpuErrchk(cudaDeviceSynchronize());
	cudaFree(feat_tr);
	cudaFree(kern_tr);
	logger.caller_state = 4; //Inv transform exec state.

#if DEBUG
	printf("\nPrinting EWM result\n");
	print_3d_tensor_float(ewm_res, tileSize*tileSize, feat_tiles_per_ch_horiz, feat_tiles_per_ch_vert*K);
#endif

	/*
	gpuErrchk(cudaMallocManaged(&conv_out, sizeof(float)*feat_tiles_per_ch*m*m*C*K));
	int THREADS_INV = TPB;//8;
	int BLOCKS_V = (feat_tiles_per_ch_vert + THREADS_INV - 1)/THREADS_INV;
	int BLOCKS_U = (feat_tiles_per_ch_horiz + THREADS_INV - 1)/THREADS_INV;
	int BLOCKS_W = (K + THREADS_INV - 1)/THREADS_INV;
	dim3 threads_inv(THREADS_MUL, THREADS_MUL, THREADS_MUL);
	dim3 blocks_inv(BLOCKS_U, BLOCKS_V, BLOCKS_W);
	inverse_transform<<<blocks_inv, threads_inv>>>(ewm_res, conv_out, H, K, m);
	gpuErrchk(cudaDeviceSynchronize());
	logger.caller_state = 5; //Finished exec state
	*/
	std::thread threadKill(&NVMLMonThread::killThread, &logger);
	threadStart.join();
	threadKill.join();

	//printf("\nPrinting partial EWM output for sanity check\n");
	//print_3d_tensor_float(ewm_res, m*m*2, 1, 1);

	cudaFree(ewm_res);
	//cudaFree(conv_out);
	
	printf("\n Finished... \n");
	return 0;
}

