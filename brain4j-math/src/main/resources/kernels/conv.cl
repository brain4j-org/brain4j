__kernel void convolve1d_direct(
    __global const float* in,
    __global const float* kern,
    __global float* out,
    int inputSize,
    int kernelSize,
    int outputSize,
    int paddingLeft)
{
    int i = get_global_id(0);
    if (i >= outputSize) return;
    
    float sum = 0.0f;
    
    __local float localKernel[256];
    int lid = get_local_id(0);
    if (lid < kernelSize) {
        localKernel[lid] = kern[kernelSize - 1 - lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int k = 0; k < kernelSize; k++) {
        int pos = i - paddingLeft + k;
        if (pos >= 0 && pos < inputSize) {
            sum += in[pos] * localKernel[k];
        }
    }
    
    out[i] = sum;
}

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

typedef struct Complex {
    float real;
    float imag;
} Complex;

inline Complex complex_mul(Complex a, Complex b) {
    Complex res;
    res.real = a.real * b.real - a.imag * b.imag;
    res.imag = a.real * b.imag + a.imag * b.real;
    return res;
}

inline Complex twiddle_factor(int k, int n, int direction) {
    Complex w;
    float angle = direction * 2.0f * M_PI * k / n;
    w.real = cos(angle);
    w.imag = sin(angle);
    return w;
}

inline int bit_reverse(int x, int bits) {
    int r = 0;
    for (int i = 0; i < bits; i++) {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    return r;
}

__kernel void fft1d(
    __global Complex* data,
    int n,
    int direction,
    int log2n)
{
    int gid = get_global_id(0);
    if (gid >= n / 2) return;
    
    __local Complex localData[256];
    int lid = get_local_id(0);
    
    int rev_gid = bit_reverse(gid, log2n);
    if (gid < rev_gid) {
        Complex temp = data[gid];
        data[gid] = data[rev_gid];
        data[rev_gid] = temp;
    }
    
    barrier(CLK_GLOBAL_MEM_FENCE);
    
    for (int stride = 1; stride < n; stride *= 2) {
        int butterflySize = stride * 2;
        
        Complex w = twiddle_factor(gid % stride, butterflySize, direction);
        
        for (int i = gid * butterflySize; i < n; i += get_global_size(0) * butterflySize) {
            if (i + stride < n) {
                int idx1 = i;
                int idx2 = i + stride;
                
                Complex temp = complex_mul(w, data[idx2]);
                
                Complex t1;
                t1.real = data[idx1].real + temp.real;
                t1.imag = data[idx1].imag + temp.imag;
                
                Complex t2;
                t2.real = data[idx1].real - temp.real;
                t2.imag = data[idx1].imag - temp.imag;
                
                data[idx1] = t1;
                data[idx2] = t2;
            }
        }
        
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
    
    if (direction == -1) {
        data[gid].real /= n;
        data[gid].imag /= n;
    }
}

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

__kernel void complex_pointwise_mul(
    __global const Complex* a,
    __global const Complex* b,
    __global Complex* c,
    int size)
{
    int i = get_global_id(0);
    if (i >= size) return;
    
    c[i] = complex_mul(a[i], b[i]);
}

__kernel void convolve1d_fft(
    __global const float* in,
    __global const float* kern,
    __global float* out,
    __global Complex* buffer1,
    __global Complex* buffer2,
    __global Complex* buffer3,
    int inputSize,
    int kernelSize,
    int fftSize,
    int outputSize,
    int paddingLeft)
{
    int i = get_global_id(0);
    
    if (i < fftSize) {
        buffer1[i].real = (i < inputSize) ? in[i] : 0.0f;
        buffer1[i].imag = 0.0f;
        
        buffer2[i].real = (i < kernelSize) ? kern[kernelSize - 1 - i] : 0.0f;
        buffer2[i].imag = 0.0f;
    }
    
    barrier(CLK_GLOBAL_MEM_FENCE);
    
    if (i == 0) {
        for (int stride = 1; stride < fftSize; stride *= 2) {
            int butterflySize = stride * 2;
            for (int j = 0; j < fftSize; j += butterflySize) {
                for (int k = 0; k < stride; k++) {
                    if (j + k + stride < fftSize) {
                        int idx1 = j + k;
                        int idx2 = j + k + stride;
                        
                        Complex w = twiddle_factor(k, butterflySize, 1);
                        Complex temp = complex_mul(w, buffer1[idx2]);
                        
                        Complex t1;
                        t1.real = buffer1[idx1].real + temp.real;
                        t1.imag = buffer1[idx1].imag + temp.imag;
                        
                        Complex t2;
                        t2.real = buffer1[idx1].real - temp.real;
                        t2.imag = buffer1[idx1].imag - temp.imag;
                        
                        buffer1[idx1] = t1;
                        buffer1[idx2] = t2;
                    }
                }
            }
        }
        
        for (int stride = 1; stride < fftSize; stride *= 2) {
            int butterflySize = stride * 2;
            for (int j = 0; j < fftSize; j += butterflySize) {
                for (int k = 0; k < stride; k++) {
                    if (j + k + stride < fftSize) {
                        int idx1 = j + k;
                        int idx2 = j + k + stride;
                        
                        Complex w = twiddle_factor(k, butterflySize, 1);
                        Complex temp = complex_mul(w, buffer2[idx2]);
                        
                        Complex t1;
                        t1.real = buffer2[idx1].real + temp.real;
                        t1.imag = buffer2[idx1].imag + temp.imag;
                        
                        Complex t2;
                        t2.real = buffer2[idx1].real - temp.real;
                        t2.imag = buffer2[idx1].imag - temp.imag;
                        
                        buffer2[idx1] = t1;
                        buffer2[idx2] = t2;
                    }
                }
            }
        }
        
        for (int j = 0; j < fftSize; j++) {
            buffer3[j] = complex_mul(buffer1[j], buffer2[j]);
        }
        
        for (int stride = 1; stride < fftSize; stride *= 2) {
            int butterflySize = stride * 2;
            for (int j = 0; j < fftSize; j += butterflySize) {
                for (int k = 0; k < stride; k++) {
                    if (j + k + stride < fftSize) {
                        int idx1 = j + k;
                        int idx2 = j + k + stride;
                        
                        Complex w = twiddle_factor(k, butterflySize, -1);
                        Complex temp = complex_mul(w, buffer3[idx2]);
                        
                        Complex t1;
                        t1.real = buffer3[idx1].real + temp.real;
                        t1.imag = buffer3[idx1].imag + temp.imag;
                        
                        Complex t2;
                        t2.real = buffer3[idx1].real - temp.real;
                        t2.imag = buffer3[idx1].imag - temp.imag;
                        
                        buffer3[idx1] = t1;
                        buffer3[idx2] = t2;
                    }
                }
            }
        }
        
        for (int j = 0; j < fftSize; j++) {
            buffer3[j].real /= fftSize;
            buffer3[j].imag /= fftSize;
        }
    }
    
    barrier(CLK_GLOBAL_MEM_FENCE);
    
    if (i < outputSize) {
        int startIdx;
        if (paddingLeft == 0) {
            startIdx = kernelSize - 1;
        } else if (paddingLeft == (kernelSize - 1) / 2) {
            startIdx = paddingLeft;
        } else {
            startIdx = 0;
        }
        
        float result = buffer3[startIdx + i].real;
        if (fabs(result) < 1e-10f) {
            result = 0.0f;
        }
        out[i] = result;
    }
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