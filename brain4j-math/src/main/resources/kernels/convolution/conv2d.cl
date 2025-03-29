#include "../common_definitions.cl"

__kernel void convolve2d_direct(
    __global const float* in,
    __global const float* kern,
    __global float* out,
    int inputRows,
    int inputCols,
    int kernelRows,
    int kernelCols,
    int outputRows,
    int outputCols,
    int paddingTop,
    int paddingLeft)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    
    if (row >= outputRows || col >= outputCols) return;
    
    float sum = 0.0f;
    
    __local float localKernel[256];
    int lid = get_local_id(0) * get_local_size(1) + get_local_id(1);
    if (lid < kernelRows * kernelCols) {
        int kr = kernelRows - 1 - (lid / kernelCols);
        int kc = kernelCols - 1 - (lid % kernelCols);
        localKernel[lid] = kern[kr * kernelCols + kc];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int ki = 0; ki < kernelRows; ki++) {
        for (int kj = 0; kj < kernelCols; kj++) {
            int r = row - paddingTop + ki;
            int c = col - paddingLeft + kj;
            
            if (r >= 0 && r < inputRows && c >= 0 && c < inputCols) {
                int kernelIdx = ki * kernelCols + kj;
                sum += in[r * inputCols + c] * localKernel[kernelIdx];
            }
        }
    }
    
    out[row * outputCols + col] = sum;
}

__kernel void convolve2d_fft_extract(
    __global const Complex* result,
    __global float* out,
    int outputRows,
    int outputCols,
    int startRow,
    int startCol,
    int resultCols)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    
    if (row >= outputRows || col >= outputCols) return;
    
    float val = result[(startRow + row) * resultCols + (startCol + col)].real;
    if (fabs(val) < 1e-10f) {
        val = 0.0f;
    }
    
    out[row * outputCols + col] = val;
}  