struct Complex
{
    float real;
    float imag;
};

static const float M_PI = 3.14159265358979323846f; // TODO: move this to common

Complex complex_mul(Complex a, Complex b)
{
    Complex res;
    res.real = a.real * b.real - a.imag * b.imag;
    res.imag = a.real * b.imag + a.imag * b.real;
    return res;
}

Complex complex_add(Complex a, Complex b)
{
    Complex res;
    res.real = a.real + b.real;
    res.imag = a.imag + b.imag;
    return res;
}

Complex complex_sub(Complex a, Complex b)
{
    Complex res;
    res.real = a.real - b.real;
    res.imag = a.imag - b.imag;
    return res;
}

Complex complex_conj(Complex a)
{
    Complex res;
    res.real = a.real;
    res.imag = -a.imag;
    return res;
}

float complex_abs(Complex a)
{
    return sqrt(a.real * a.real + a.imag * a.imag);
}

Complex twiddle_factor(int k, int n, int direction)
{
    Complex w;
    float angle = float(direction) * 2.0f * M_PI * float(k) / float(n);
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
    uniform RWStructuredBuffer<Complex> c,
    uniform int size
)
{
    int i = dispatchThreadID.x;
    if (i >= size) return;

    c[i] = complex_mul(a[i], b[i]);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void complex_pointwise_add(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<Complex> a,
    uniform StructuredBuffer<Complex> b,
    uniform RWStructuredBuffer<Complex> c,
    uniform int size
)
{
    int i = dispatchThreadID.x;
    if (i >= size) return;

    c[i] = complex_add(a[i], b[i]);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void real_to_complex(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> input,
    uniform RWStructuredBuffer<Complex> output,
    uniform int size,
    uniform int inputSize
)
{
    int i = dispatchThreadID.x;
    if (i >= size) return;

    if (i < inputSize) {
        output[i].real = input[i];
        output[i].imag = 0.0f;
    } else {
        output[i].real = 0.0f;
        output[i].imag = 0.0f;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void complex_to_real(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<Complex> input,
    uniform RWStructuredBuffer<float> output,
    uniform int size
)
{
    int i = dispatchThreadID.x;
    if (i >= size) return;

    output[i] = input[i].real;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void complex_magnitude(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<Complex> input,
    uniform RWStructuredBuffer<float> output,
    uniform int size
)
{
    int i = dispatchThreadID.x;
    if (i >= size) return;

    output[i] = complex_abs(input[i]);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void complex_conjugate(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform RWStructuredBuffer<Complex> data,
    uniform int size
)
{
    int i = dispatchThreadID.x;
    if (i >= size) return;

    data[i].imag = -data[i].imag;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void apply_twiddle(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform RWStructuredBuffer<Complex> data,
    uniform int stride,
    uniform int butterflySize,
    uniform int direction,
    uniform int size
)
{
    int i = dispatchThreadID.x;
    if (i >= size) return;

    int j = (i / stride) * butterflySize + (i % stride);
    int k = i % stride;

    if (j + stride < size) {
        Complex w = twiddle_factor(k, butterflySize, direction);
        Complex temp = complex_mul(w, data[j + stride]);

        Complex t1;
        t1.real = data[j].real + temp.real;
        t1.imag = data[j].imag + temp.imag;

        Complex t2;
        t2.real = data[j].real - temp.real;
        t2.imag = data[j].imag - temp.imag;

        data[j] = t1;
        data[j + stride] = t2;
    }
}

