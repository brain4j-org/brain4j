__kernel void matmul(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int m,
    const int n,
    const int p
) {
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    
    const int local_row = get_local_id(0);
    const int local_col = get_local_id(1);
    
    const int block_dim_x = get_local_size(0);
    const int block_dim_y = get_local_size(1);
    
    if (row < m && col < p) {
        float sum = 0.0f;
        
        int k = 0;
        
        for (; k + 3 < n; k += 4) {
            sum += A[row * n + k] * B[k * p + col];
            sum += A[row * n + k + 1] * B[(k + 1) * p + col];
            sum += A[row * n + k + 2] * B[(k + 2) * p + col];
            sum += A[row * n + k + 3] * B[(k + 3) * p + col];
        }
        
        for (; k < n; k++) {
            sum += A[row * n + k] * B[k * p + col];
        }
        
        C[row * p + col] = sum;
    }
}

__kernel void element_wise_add(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int size
) {
    const int i = get_global_id(0);
    
    if (i + 3 < size && (i % 4) == 0) {
        float4 a_vec = vload4(i/4, A);
        float4 b_vec = vload4(i/4, B);
        
        float4 c_vec = a_vec + b_vec;
        
        vstore4(c_vec, i/4, C);
    }
    else if (i < size) {
        C[i] = A[i] + B[i];
    }
}

__kernel void element_wise_mul(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int size
) {
    const int i = get_global_id(0);
    
    if (i + 3 < size && (i % 4) == 0) {
        float4 a_vec = vload4(i/4, A);
        float4 b_vec = vload4(i/4, B);
        
        float4 c_vec = a_vec * b_vec;
        
        vstore4(c_vec, i/4, C);
    }
    else if (i < size) {
        C[i] = A[i] * B[i];
    }
}

__kernel void convolve1d(
    __global const float* input,
    __global const float* kernel,
    __global float* output,
    const int inputSize,
    const int kernelSize,
    const int outputSize, 
    const int stride,
    const int padding
) {
    const int i = get_global_id(0);
    
    if (i < outputSize) {
        float sum = 0.0f;
        int inputPos = i * stride - padding;
        
        int k = 0;
        int kStart = max(0, -inputPos);
        int kEnd = min(kernelSize, inputSize - inputPos);
        
        for (; k + 3 < kEnd; k += 4) {
            if (inputPos + k >= 0 && inputPos + k + 3 < inputSize) {
                float4 inputVec = (float4)(
                    input[inputPos + k],
                    input[inputPos + k + 1],
                    input[inputPos + k + 2],
                    input[inputPos + k + 3]
                );
                
                float4 kernelVec = (float4)(
                    kernel[kernelSize - 1 - k],
                    kernel[kernelSize - 1 - (k + 1)],
                    kernel[kernelSize - 1 - (k + 2)],
                    kernel[kernelSize - 1 - (k + 3)]
                );
                
                float4 result = inputVec * kernelVec;
                sum += result.x + result.y + result.z + result.w;
            } else {
                for (int j = 0; j < 4; j++) {
                    int pos = inputPos + k + j;
                    if (pos >= 0 && pos < inputSize) {
                        sum += input[pos] * kernel[kernelSize - 1 - (k + j)];
                    }
                }
            }
        }
        
        for (; k < kEnd; k++) {
            sum += input[inputPos + k] * kernel[kernelSize - 1 - k];
        }
        
        output[i] = sum;
    }
}

__kernel void convolve2d(
    __global const float* input,
    __global const float* kernel,
    __global float* output,
    const int inputRows,
    const int inputCols,
    const int kernelRows,
    const int kernelCols,
    const int outputRows,
    const int outputCols,
    const int stride,
    const int padding
) {
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    
    if (row < outputRows && col < outputCols) {
        float sum = 0.0f;
        
        #define LOCAL_SIZE 16
        
        for (int ki = 0; ki < kernelRows; ki++) {
            int inputRowPos = row * stride + ki - padding;
            
            if (inputRowPos >= 0 && inputRowPos < inputRows) {
                int inputRowOffset = inputRowPos * inputCols;
                int kernelRowOffset = (kernelRows - 1 - ki) * kernelCols;
                
                int kj = 0;
                
                for (; kj + 3 < kernelCols; kj += 4) {
                    int colPos0 = col * stride + kj - padding;
                    int colPos1 = col * stride + kj + 1 - padding;
                    int colPos2 = col * stride + kj + 2 - padding;
                    int colPos3 = col * stride + kj + 3 - padding;
                    
                    bool validCol0 = colPos0 >= 0 && colPos0 < inputCols;
                    bool validCol1 = colPos1 >= 0 && colPos1 < inputCols;
                    bool validCol2 = colPos2 >= 0 && colPos2 < inputCols;
                    bool validCol3 = colPos3 >= 0 && colPos3 < inputCols;
                    
                    if (validCol0) sum += input[inputRowOffset + colPos0] * 
                                         kernel[kernelRowOffset + (kernelCols - 1 - kj)];
                    if (validCol1) sum += input[inputRowOffset + colPos1] * 
                                         kernel[kernelRowOffset + (kernelCols - 1 - (kj + 1))];
                    if (validCol2) sum += input[inputRowOffset + colPos2] * 
                                         kernel[kernelRowOffset + (kernelCols - 1 - (kj + 2))];
                    if (validCol3) sum += input[inputRowOffset + colPos3] * 
                                         kernel[kernelRowOffset + (kernelCols - 1 - (kj + 3))];
                }
                
                for (; kj < kernelCols; kj++) {
                    int inputColPos = col * stride + kj - padding;
                    
                    if (inputColPos >= 0 && inputColPos < inputCols) {
                        sum += input[inputRowOffset + inputColPos] * 
                              kernel[kernelRowOffset + (kernelCols - 1 - kj)];
                    }
                }
            }
        }
        
        output[row * outputCols + col] = sum;
    }
}