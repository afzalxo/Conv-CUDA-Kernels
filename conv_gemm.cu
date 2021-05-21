










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

