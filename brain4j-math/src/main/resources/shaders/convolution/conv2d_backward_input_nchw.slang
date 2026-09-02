static const int TILE_SIZE = 16;
static const int MAX_KERNEL = 9;
static const int MAX_INPUT_TILE = TILE_SIZE * 4 + MAX_KERNEL - 1;

groupshared float Tile[MAX_INPUT_TILE][MAX_INPUT_TILE + 1];

[shader("compute")]
[numthreads(16, 16, 1)]
void conv2d_backward_input_nchw(
    uint3 id : SV_DispatchThreadID,
    uniform StructuredBuffer<float> gradOutput,
    uniform StructuredBuffer<float> kernel,
    uniform RWStructuredBuffer<float> gradInput,
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
    int inX = id.x;
    int combinedY = id.y;
    int b = id.z;
    if (inX >= inWidth || b >= batchSize) return;

    int c = combinedY / inHeight;
    int inY = combinedY % inHeight;
    if (c >= inChannels || inY >= inHeight) return;

    float sum = 0.0f;
    for (int f = 0; f < numFilters; f++) {
        for (int ky = 0; ky < kernelHeight; ky++) {
            int oyNumerator = inY - ky;
            if (oyNumerator < 0 || (oyNumerator % stride) != 0) continue;
            int oy = oyNumerator / stride;
            if (oy >= outHeight) continue;

            for (int kx = 0; kx < kernelWidth; kx++) {
                int oxNumerator = inX - kx;
                if (oxNumerator < 0 || (oxNumerator % stride) != 0) continue;
                int ox = oxNumerator / stride;
                if (ox >= outWidth) continue;

                int gradOutIdx = ((b * numFilters + f) * outHeight + oy) * outWidth + ox;
                int kernelIdx = ((f * inChannels + c) * kernelHeight + ky) * kernelWidth + kx;
                sum += gradOutput[gradOutIdx] * kernel[kernelIdx];
            }
        }
    }

    gradInput[((b * inChannels + c) * inHeight + inY) * inWidth + inX] = sum;
}
