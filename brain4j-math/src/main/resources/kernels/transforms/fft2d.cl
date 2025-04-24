#include "../common_definitions.cl"
#include "../complex/complex_ops.cl"

__kernel void fft2d(
    __global Complex* input,
    __global Complex* output,
    int rows,
    int cols,
    int direction)
{
    int row = get_global_id(0);
    if (row >= rows) return;
    
    __local Complex row_data[256];
    int lid = get_local_id(1);
    
    if (lid < cols) {
        row_data[lid] = input[row * cols + lid];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int log2cols = 0;
    for (int i = cols; i > 1; i >>= 1) log2cols++;
    
    for (int stride = 1; stride < cols; stride *= 2) {
        int butterflySize = stride * 2;
        
        Complex w = twiddle_factor(lid % stride, butterflySize, direction);
        
        for (int i = lid * butterflySize; i < cols; i += get_local_size(1) * butterflySize) {
            if (i + stride < cols) {
                int idx1 = i;
                int idx2 = i + stride;
                
                Complex temp = complex_mul(w, row_data[idx2]);
                
                Complex t1;
                t1.real = row_data[idx1].real + temp.real;
                t1.imag = row_data[idx1].imag + temp.imag;
                
                Complex t2;
                t2.real = row_data[idx1].real - temp.real;
                t2.imag = row_data[idx1].imag - temp.imag;
                
                row_data[idx1] = t1;
                row_data[idx2] = t2;
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (lid < cols) {
        output[row * cols + lid] = row_data[lid];
    }
}

__kernel void fft2d_transpose(
    __global Complex* input,
    __global Complex* output,
    int rows,
    int cols)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    
    if (row >= rows || col >= cols) return;
    
    output[col * rows + row] = input[row * cols + col];
}