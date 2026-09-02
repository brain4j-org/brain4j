static const int TILE_SIZE = 16;
static const int MAX_KERNEL = 9;
static const int MAX_INPUT_TILE = TILE_SIZE * 4 + MAX_KERNEL - 1;

groupshared float Tile[MAX_INPUT_TILE][MAX_INPUT_TILE + 1];

[shader("compute")]
[numthreads(256, 1, 1)]
void convolve1d_fft(
    uint3 id : SV_DispatchThreadID,
    uniform StructuredBuffer<float> input,
    uniform StructuredBuffer<float> kernel,
    uniform RWStructuredBuffer<float> output,
    uniform RWStructuredBuffer<float> buffer1,
    uniform RWStructuredBuffer<float> buffer2,
    uniform RWStructuredBuffer<float> buffer3,
    uniform int inputSize,
    uniform int kernelSize,
    uniform int fftSize,
    uniform int outputSize,
    uniform int paddingLeft
)
{
    int outIdx = id.x;
    if (outIdx >= outputSize) return;

    float sum = 0.0f;
    for (int k = 0; k < kernelSize; k++) {
        int inIdx = outIdx + k - paddingLeft;
        if (inIdx >= 0 && inIdx < inputSize) {
            sum += input[inIdx] * kernel[k];
        }
    }
    output[outIdx] = sum;
}
