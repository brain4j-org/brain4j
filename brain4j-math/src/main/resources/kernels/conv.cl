__kernel void convolve1d(
    __global const float* in,
    __global const float* kern,
    __global float* out,
    int inputSize,
    int kernelSize,
    int outputSize,
    int stride,
    int padding)
{
    int i = get_global_id(0);
    if (i >= outputSize) return;
    
    float sum = 0.0f;
    int inputPos = i * stride - padding;
    
    for (int k = 0; k < kernelSize; k++) {
        int pos = inputPos + k;
        if (pos >= 0 && pos < inputSize) {
            sum += in[pos] * kern[k];
        }
    }
    
    out[i] = sum;
}

__kernel void convolve2d(
    __global const float* in,
    __global const float* kern,
    __global float* out,
    int inputRows,
    int inputCols,
    int kernelRows,
    int kernelCols,
    int outputRows,
    int outputCols,
    int stride,
    int paddingRows,
    int paddingCols)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    
    if (row >= outputRows || col >= outputCols) return;
    
    float sum = 0.0f;
    
    for (int ki = 0; ki < kernelRows; ki++) {
        for (int kj = 0; kj < kernelCols; kj++) {
            int r = row * stride + ki - paddingRows;
            int c = col * stride + kj - paddingCols;
            
            if (r >= 0 && r < inputRows && c >= 0 && c < inputCols) {
                sum += in[r * inputCols + c] * kern[ki * kernelCols + kj];
            }
        }
    }
    
    out[row * outputCols + col] = sum;
} 