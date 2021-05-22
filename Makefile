conv_gemm:
	nvcc conv_gemm.cu -o conv_gemm -lcuda -lnvidia-ml
gemm_kernel:
	nvcc gemm_kernel.cu -o gemm_kernel -lcuda -lnvidia-ml

clean_conv:
	rm conv_gemm
clean_gemm:
	rm gemm_kernel
