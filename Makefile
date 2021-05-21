all:
	nvcc gemm_kernel.cu -o gemm_kernel -lcuda -lnvidia-ml

clean:
	rm gemm_kernel
