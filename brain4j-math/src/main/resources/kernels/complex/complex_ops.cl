#include "../common_definitions.cl"

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