static const int TILE_SIZE = 16;
static const int MAX_KERNEL = 9;
static const int MAX_INPUT_TILE = TILE_SIZE * 4 + MAX_KERNEL - 1;

groupshared float Tile[MAX_INPUT_TILE][MAX_INPUT_TILE + 1];

[shader("compute")]
[numthreads(16, 16, 1)]
void conv2d_nchw_shared(
    uint3 groupID      : SV_GroupID,
    uint3 localID      : SV_GroupThreadID,
    uint3 globalID     : SV_DispatchThreadID,
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
    int groupX = groupID.x;
    int groupY = groupID.y;
    int groupZ = groupID.z;
    int localX = localID.x;
    int localY = localID.y;
    int b = groupZ / numFilters;
    int f = groupZ % numFilters;

    if (b >= batchSize || f >= numFilters) return;

    int outX = groupX * TILE_SIZE + localX;
    int outY = groupY * TILE_SIZE + localY;
    float sum = 0.0f;

    int tileStartX = groupX * TILE_SIZE * stride;
    int tileStartY = groupY * TILE_SIZE * stride;
    int tileWidth = TILE_SIZE * stride + kernelWidth - 1;
    int tileHeight = TILE_SIZE * stride + kernelHeight - 1;

    for (int c = 0; c < inChannels; c++) {
        for (int y = localY; y < tileHeight; y += TILE_SIZE) {
            for (int x = localX; x < tileWidth; x += TILE_SIZE) {
                int globalX = tileStartX + x;
                int globalY = tileStartY + y;
                float value = 0.0f;

                if (globalX < inWidth && globalY < inHeight) {
                    int idx = ((b * inChannels + c) * inHeight + globalY) * inWidth + globalX;
                    value = input[idx];
                }

                Tile[y][x] = value;
            }
        }

        GroupMemoryBarrierWithGroupSync();

        if (outX < outWidth && outY < outHeight) {
            int inTileX = localX * stride;
            int inTileY = localY * stride;

            for (int ky = 0; ky < kernelHeight; ky++) {
                for (int kx = 0; kx < kernelWidth; kx++) {
                    int kernelIdx = ((f * inChannels + c) * kernelHeight + ky) * kernelWidth + kx;
                    sum += Tile[inTileY + ky][inTileX + kx] * kernel[kernelIdx];
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (outX < outWidth && outY < outHeight) {
        int outputIdx = ((b * numFilters + f) * outHeight + outY) * outWidth + outX;
        output[outputIdx] = sum;
    }
}
