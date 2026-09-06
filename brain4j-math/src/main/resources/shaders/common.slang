struct Complex
{
    float real;
    float imag;
};

static const float M_PI = 3.14159265358979323846f;
static const float M_E = 2.71828182845904523536f;
static const float M_SQRT2 = 1.41421356237309504880f;
static const float M_SQRT1_2 = 0.70710678118654752440f;

static const int TILE_SIZE = 16;
static const int MAX_RANK = 16;

static const int FA_TILE_SIZE = 16;
static const int FA_HEAD_DIM = 64;

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
    float angle = direction * 2.0f * M_PI * float(k) / float(n);
    w.real = cos(angle);
    w.imag = sin(angle);
    return w;
}

float safe_exp(float x)
{
    return exp(clamp(x, -88.0f, 88.0f));
}

float safe_log(float x)
{
    return log(max(x, 1e-10f));
}
