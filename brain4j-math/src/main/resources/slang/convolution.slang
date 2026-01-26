import complex_ops;

groupshared float localKernel1D[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void convolve1d_direct(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uint3 groupThreadID : SV_GroupThreadID,
    uniform StructuredBuffer<float> input,
    uniform StructuredBuffer<float> kern,
    uniform RWStructuredBuffer<float> output,
    uniform int inputSize,
    uniform int kernelSize,
    uniform int outputSize,
    uniform int paddingLeft
)
{
    int i = dispatchThreadID.x;
    if (i >= outputSize) {
        return;
    }

    int lid = groupThreadID.x;

    // we load the kernel into local memory (reversed for convolution)
    if (lid < kernelSize) {
        localKernel1D[lid] = kern[kernelSize - 1 - lid];
    }
    GroupMemoryBarrierWithGroupSync();

    float sum = 0.0f;
    for (int k = 0; k < kernelSize; k++) {
        int pos = i - paddingLeft + k;
        if (pos >= 0 && pos < inputSize) {
            sum += input[pos] * localKernel1D[k];
        }
    }

    output[i] = sum;
}

groupshared float localKernel2D[256];

[shader("compute")]
[numthreads(16, 16, 1)]
void convolve2d_direct(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uint3 groupThreadID : SV_GroupThreadID,
    uniform StructuredBuffer<float> input,
    uniform StructuredBuffer<float> kern,
    uniform RWStructuredBuffer<float> output,
    uniform int inputRows,
    uniform int inputCols,
    uniform int kernelRows,
    uniform int kernelCols,
    uniform int outputRows,
    uniform int outputCols,
    uniform int paddingTop,
    uniform int paddingLeft
)
{
    int row = dispatchThreadID.x;
    int col = dispatchThreadID.y;

    if (row >= outputRows || col >= outputCols) return;

    int lid = groupThreadID.x * 16 + groupThreadID.y;

    // we load the kernel into local memory (reversed for convolution)
    if (lid < kernelRows * kernelCols) {
        int kr = kernelRows - 1 - (lid / kernelCols);
        int kc = kernelCols - 1 - (lid % kernelCols);
        localKernel2D[lid] = kern[kr * kernelCols + kc];
    }
    GroupMemoryBarrierWithGroupSync();

    float sum = 0.0f;
    for (int ki = 0; ki < kernelRows; ki++) {
        for (int kj = 0; kj < kernelCols; kj++) {
            int r = row - paddingTop + ki;
            int c = col - paddingLeft + kj;

            if (r >= 0 && r < inputRows && c >= 0 && c < inputCols) {
                int kernelIdx = ki * kernelCols + kj;
                sum += input[r * inputCols + c] * localKernel2D[kernelIdx];
            }
        }
    }

    output[row * outputCols + col] = sum;
}

/*
struct Complex
{
    float real;
    float imag;
};
*/

[shader("compute")]
[numthreads(16, 16, 1)]
void convolve2d_fft_extract(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<Complex> result,
    uniform RWStructuredBuffer<float> output,
    uniform int outputRows,
    uniform int outputCols,
    uniform int startRow,
    uniform int startCol,
    uniform int resultCols
)
{
    int row = dispatchThreadID.x;
    int col = dispatchThreadID.y;

    if (row >= outputRows || col >= outputCols) return;

    float val = result[(startRow + row) * resultCols + (startCol + col)].real;

    // let's clean up some noise
    if (abs(val) < 1e-10f) {
        val = 0.0f;
    }

    output[row * outputCols + col] = val;
}

Complex conv_complex_mul(Complex a, Complex b)
{
    Complex res;
    res.real = a.real * b.real - a.imag * b.imag;
    res.imag = a.real * b.imag + a.imag * b.real;
    return res;
}

Complex conv_twiddle_factor(int k, int n, int direction)
{
    Complex w;
    float angle = float(direction) * 2.0f * 3.14159265358979323846f /*M_PI*/ * float(k) / float(n);
    w.real = cos(angle);
    w.imag = sin(angle);
    return w;
}
[shader("compute")]
[numthreads(256, 1, 1)]
void convolve1d_fft_prepare(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> input,
    uniform StructuredBuffer<float> kern,
    uniform RWStructuredBuffer<Complex> inputFFT,
    uniform RWStructuredBuffer<Complex> kernFFT,
    uniform int inputSize,
    uniform int kernelSize,
    uniform int fftSize
)
{
    int i = dispatchThreadID.x;
    if (i >= fftSize) return;

    // zero-padded input buffer
    inputFFT[i].real = (i < inputSize) ? input[i] : 0.0f;
    inputFFT[i].imag = 0.0f;
    kernFFT[i].real = (i < kernelSize) ? kern[kernelSize - 1 - i] : 0.0f;
    kernFFT[i].imag = 0.0f;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void convolve1d_fft_multiply(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<Complex> inputFFT,
    uniform StructuredBuffer<Complex> kernFFT,
    uniform RWStructuredBuffer<Complex> resultFFT,
    uniform int fftSize
)
{
    int i = dispatchThreadID.x;
    if (i >= fftSize) return;

    resultFFT[i] = conv_complex_mul(inputFFT[i], kernFFT[i]);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void convolve1d_fft_extract(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<Complex> resultFFT,
    uniform RWStructuredBuffer<float> output,
    uniform int outputSize,
    uniform int kernelSize,
    uniform int paddingLeft
)
{
    int i = dispatchThreadID.x;
    if (i >= outputSize) return;

    // here we compute the start index from the padding value
    int startIdx;
    if (paddingLeft == 0) {
        startIdx = kernelSize - 1;
    } else if (paddingLeft == (kernelSize - 1) / 2) {
        startIdx = paddingLeft;
    } else {
        startIdx = 0;
    }

    float result = resultFFT[startIdx + i].real;

    // noise cleaning
    if (abs(result) < 1e-10f) {
        result = 0.0f;
    }

    output[i] = result;
}
[shader("compute")]
[numthreads(16, 16, 1)]
void convolve2d_fft_prepare_input(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> input,
    uniform RWStructuredBuffer<Complex> inputFFT,
    uniform int inputRows,
    uniform int inputCols,
    uniform int fftRows,
    uniform int fftCols
)
{
    int row = dispatchThreadID.x;
    int col = dispatchThreadID.y;

    if (row >= fftRows || col >= fftCols) return;

    int fftIdx = row * fftCols + col;

    if (row < inputRows && col < inputCols) {
        inputFFT[fftIdx].real = input[row * inputCols + col];
    } else {
        inputFFT[fftIdx].real = 0.0f;
    }
    inputFFT[fftIdx].imag = 0.0f;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void convolve2d_fft_prepare_kernel(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> kern,
    uniform RWStructuredBuffer<Complex> kernFFT,
    uniform int kernelRows,
    uniform int kernelCols,
    uniform int fftRows,
    uniform int fftCols
)
{
    int row = dispatchThreadID.x;
    int col = dispatchThreadID.y;

    if (row >= fftRows || col >= fftCols) return;

    int fftIdx = row * fftCols + col;

    if (row < kernelRows && col < kernelCols) {
        int kr = kernelRows - 1 - row;
        int kc = kernelCols - 1 - col;
        kernFFT[fftIdx].real = kern[kr * kernelCols + kc];
    } else {
        kernFFT[fftIdx].real = 0.0f;
    }
    kernFFT[fftIdx].imag = 0.0f;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void convolve2d_fft_multiply(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<Complex> inputFFT,
    uniform StructuredBuffer<Complex> kernFFT,
    uniform RWStructuredBuffer<Complex> resultFFT,
    uniform int fftRows,
    uniform int fftCols
)
{
    int row = dispatchThreadID.x;
    int col = dispatchThreadID.y;

    if (row >= fftRows || col >= fftCols) return;

    int idx = row * fftCols + col;
    resultFFT[idx] = conv_complex_mul(inputFFT[idx], kernFFT[idx]);
}
