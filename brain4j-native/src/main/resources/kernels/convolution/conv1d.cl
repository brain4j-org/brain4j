#include "../common_definitions.cl"
#include "../complex/complex_ops.cl"

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