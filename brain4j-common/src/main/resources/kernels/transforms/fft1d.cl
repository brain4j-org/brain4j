#include "../common_definitions.cl"
#include "../complex/complex_ops.cl"
#include "bit_operations.cl"

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