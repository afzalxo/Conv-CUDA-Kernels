#ifndef GEMM_KERNEL_H
#define GEMM_KERNEL_H

__global__ void matrixMul(const int *a, const int *b, int *c, int N, int M, int K);

#endif
