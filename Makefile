gemm_kernel:
	nvcc gemm_kernel.cu -o gemm_kernel -lcuda -lnvidia-ml

conv_gemm:
	nvcc conv_gemm.cu -o conv_gemm -lcuda -lnvidia-ml

conv_gemm_explicit:
	nvcc conv_gemm_explicit.cu -o conv_gemm_explicit -lcuda -lnvidia-ml

conv_wino:
	nvcc conv_wino.cu -o conv_wino -lcuda -lnvidia-ml

clean_conv:
	rm conv_gemm
clean_gemm:
	rm gemm_kernel
