static const int TILE_SIZE = 16;
static const int MAX_KERNEL = 9;
static const int MAX_INPUT_TILE = TILE_SIZE * 4 + MAX_KERNEL - 1;

groupshared float Tile[MAX_INPUT_TILE][MAX_INPUT_TILE + 1];

[shader("compute")]
[numthreads(16, 16, 1)]
void conv2d_nchw(
    uint3 id : SV_DispatchThreadID,
    uniform StructuredBuffer<float> input,
    uniform StructuredBuffer<float> kernel,
    uniform RWStructuredBuffer<float> output,
    uniform int batchSize,
    uniform int inChannels,
    uniform int inHeight,
    uniform int inWidth,
    uniform int numFilters,
    uniform int kernelHeight,
    uniform int kernelWidth,
    uniform int stride,
    uniform int outHeight,
    uniform int outWidth
)
{
    int outX = id.x;
    int combinedY = id.y;
    int b = id.z;
    if (outX >= outWidth || b >= batchSize) return;

    int f = combinedY / outHeight;
    int outY = combinedY % outHeight;
    if (f >= numFilters || outY >= outHeight) return;

    float sum = 0.0f;
    int inYBase = outY * stride;
    int inXBase = outX * stride;

    for (int c = 0; c < inChannels; c++) {
        for (int ky = 0; ky < kernelHeight; ky++) {
            int iy = inYBase + ky;
            for (int kx = 0; kx < kernelWidth; kx++) {
                int ix = inXBase + kx;
                int inputIdx = ((b * inChannels + c) * inHeight + iy) * inWidth + ix;
                int kernelIdx = ((f * inChannels + c) * kernelHeight + ky) * kernelWidth + kx;
                sum += input[inputIdx] * kernel[kernelIdx];
            }
        }
    }

    int outputIdx = ((b * numFilters + f) * outHeight + outY) * outWidth + outX;
    output[outputIdx] = sum;
}
