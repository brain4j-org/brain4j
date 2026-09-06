struct Complex
{
    float real;
    float imag;
};

static const float M_PI = 3.14159265358979323846f;

Complex complex_mul(Complex a, Complex b)
{
    Complex res;
    res.real = a.real * b.real - a.imag * b.imag;
    res.imag = a.real * b.imag + a.imag * b.real;
    return res;
}

Complex twiddle_factor(int k, int n, int direction)
{
    Complex w;
    float angle = direction * 2.0f * M_PI * float(k) / float(n);
    w.real = cos(angle);
    w.imag = sin(angle);
    return w;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void complex_pointwise_mul(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<Complex> a,
    uniform StructuredBuffer<Complex> b,
    uniform RWStructuredBuffer<Complex> out,
    uniform int length
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    out[gid] = complex_mul(a[gid], b[gid]);
}
