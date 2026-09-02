static const int TILE_SIZE = 16;
static const int MAX_KERNEL = 9;
static const int MAX_INPUT_TILE = TILE_SIZE * 4 + MAX_KERNEL - 1;

groupshared float Tile[MAX_INPUT_TILE][MAX_INPUT_TILE + 1];

[shader("compute")]
[numthreads(16, 16, 1)]
void conv2d_backward_filter_nchw(
    uint3 id : SV_DispatchThreadID,
    uniform StructuredBuffer<float> input,
    uniform StructuredBuffer<float> gradOutput,
    uniform RWStructuredBuffer<float> gradFilter,
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
    int kx = id.x;
    int combinedY = id.y;
    int f = id.z;
    if (kx >= kernelWidth || f >= numFilters) return;

    int c = combinedY / kernelHeight;
    int ky = combinedY % kernelHeight;
    if (c >= inChannels || ky >= kernelHeight) return;

    float sum = 0.0f;
    for (int b = 0; b < batchSize; b++) {
        for (int oy = 0; oy < outHeight; oy++) {
            int inY = oy * stride + ky;
            for (int ox = 0; ox < outWidth; ox++) {
                int inX = ox * stride + kx;
                int inputIdx = ((b * inChannels + c) * inHeight + inY) * inWidth + inX;
                int gradOutIdx = ((b * numFilters + f) * outHeight + oy) * outWidth + ox;
                sum += input[inputIdx] * gradOutput[gradOutIdx];
            }
        }
    }

    gradFilter[((f * inChannels + c) * kernelHeight + ky) * kernelWidth + kx] = sum;
}
