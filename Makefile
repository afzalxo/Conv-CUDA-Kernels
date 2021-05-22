conv_gemm:
	nvcc conv_gemm.cu -o conv_gemm -lcuda -lnvidia-ml
all:
	nvcc gemm_kernel.cu -o gemm_kernel -lcuda -lnvidia-ml

clean:
	rm gemm_kernel conv_gemm
