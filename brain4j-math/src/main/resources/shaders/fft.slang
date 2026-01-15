import complex_ops;

int bit_reverse(int x, int bits)
{
    int r = 0;
    for (int i = 0; i < bits; i++) {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    return r;
}

groupshared Complex fft1d_local[256];

[shader("compute")]
[numthreads(128, 1, 1)]
void fft1d(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uint3 groupThreadID : SV_GroupThreadID,
    uniform RWStructuredBuffer<Complex> data,
    uniform int n,
    uniform int direction,
    uniform int log2n
)
{
    int gid = dispatchThreadID.x;
    if (gid >= n / 2) return;

    int lid = groupThreadID.x;

    // bit reversal permutation
    int rev_gid = bit_reverse(gid, log2n);
    if (gid < rev_gid) {
        Complex temp = data[gid];
        data[gid] = data[rev_gid];
        data[rev_gid] = temp;
    }

    // DeviceMemoryBarrierWithGroupSync(); /*maybe not needed*/

    for (int stride = 1; stride < n; stride *= 2) {
        int butterflySize = stride * 2;

        Complex w = twiddle_factor(gid % stride, butterflySize, direction);

        int idx1 = (gid / stride) * butterflySize + (gid % stride);
        int idx2 = idx1 + stride;

        if (idx2 < n) {
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

        DeviceMemoryBarrierWithGroupSync();
    }

    if (direction == -1) {
        if (gid < n) {
            data[gid].real /= float(n);
            data[gid].imag /= float(n);
        }
        if (gid + n/2 < n) {
            data[gid + n/2].real /= float(n);
            data[gid + n/2].imag /= float(n);
        }
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void bit_reverse_permute(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform RWStructuredBuffer<Complex> data,
    uniform int n,
    uniform int log2n
)
{
    int gid = dispatchThreadID.x;
    if (gid >= n) return;

    int rev = bit_reverse(gid, log2n);
    if (gid < rev) {
        Complex temp = data[gid];
        data[gid] = data[rev];
        data[rev] = temp;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fft_butterfly_stage(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform RWStructuredBuffer<Complex> data,
    uniform int n,
    uniform int stride,
    uniform int direction
)
{
    int gid = dispatchThreadID.x;
    if (gid >= n / 2) return;

    int butterflySize = stride * 2;
    int k = gid % stride;
    int idx1 = (gid / stride) * butterflySize + k;
    int idx2 = idx1 + stride;

    if (idx2 < n) {
        Complex w = twiddle_factor(k, butterflySize, direction);
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

[shader("compute")]
[numthreads(256, 1, 1)]
void fft_normalize(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform RWStructuredBuffer<Complex> data,
    uniform int n
)
{
    int gid = dispatchThreadID.x;
    if (gid >= n) return;

    float invN = 1.0f / float(n);
    data[gid].real *= invN;
    data[gid].imag *= invN;
}

groupshared Complex fft2d_row_data[256];

[shader("compute")]
[numthreads(1, 256, 1)]
void fft2d_rows(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uint3 groupThreadID : SV_GroupThreadID,
    uniform StructuredBuffer<Complex> input,
    uniform RWStructuredBuffer<Complex> output,
    uniform int rows,
    uniform int cols,
    uniform int direction
)
{
    int row = dispatchThreadID.x;
    if (row >= rows) return;

    int lid = groupThreadID.y;

    // we now load the row into local memory
    if (lid < cols) {
        fft2d_row_data[lid] = input[row * cols + lid];
    }

    GroupMemoryBarrierWithGroupSync();

    int log2cols = 0;
    for (int i = cols; i > 1; i >>= 1) log2cols++;

    // FFT butterfly stages
    for (int stride = 1; stride < cols; stride *= 2) {
        int butterflySize = stride * 2;

        if (lid < cols / 2) {
            int k = lid % stride;
            int idx1 = (lid / stride) * butterflySize + k;
            int idx2 = idx1 + stride;

            if (idx2 < cols) {
                Complex w = twiddle_factor(k, butterflySize, direction);
                Complex temp = complex_mul(w, fft2d_row_data[idx2]);

                Complex t1;
                t1.real = fft2d_row_data[idx1].real + temp.real;
                t1.imag = fft2d_row_data[idx1].imag + temp.imag;

                Complex t2;
                t2.real = fft2d_row_data[idx1].real - temp.real;
                t2.imag = fft2d_row_data[idx1].imag - temp.imag;

                fft2d_row_data[idx1] = t1;
                fft2d_row_data[idx2] = t2;
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // once we have the result we write it back
    if (lid < cols) {
        Complex val = fft2d_row_data[lid];
        if (direction == -1) {
            val.real /= float(cols);
            val.imag /= float(cols);
        }
        output[row * cols + lid] = val;
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void fft2d_transpose(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<Complex> input,
    uniform RWStructuredBuffer<Complex> output,
    uniform int rows,
    uniform int cols
)
{
    int row = dispatchThreadID.x;
    int col = dispatchThreadID.y;

    if (row >= rows || col >= cols) return;

    output[col * rows + row] = input[row * cols + col];
}

[shader("compute")]
[numthreads(16, 16, 1)]
void fft2d_copy(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<Complex> input,
    uniform RWStructuredBuffer<Complex> output,
    uniform int rows,
    uniform int cols
)
{
    int row = dispatchThreadID.x;
    int col = dispatchThreadID.y;

    if (row >= rows || col >= cols) return;

    output[row * cols + col] = input[row * cols + col];
}

