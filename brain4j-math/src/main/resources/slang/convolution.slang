static const int TILE_SIZE = 16;
static const int MAX_KERNEL = 9;

static const int MAX_INPUT_TILE = TILE_SIZE * 4 + MAX_KERNEL - 1;

groupshared float Tile[MAX_INPUT_TILE][MAX_INPUT_TILE + 1];

[shader("compute")]
[numthreads(16,16,1)]
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

    if (b >= batchSize || f >= numFilters)
        return;

    int outX = groupX * TILE_SIZE + localX;
    int outY = groupY * TILE_SIZE + localY;

    float sum = 0.0f;

    int tileStartX = groupX * TILE_SIZE * stride;
    int tileStartY = groupY * TILE_SIZE * stride;

    int tileWidth  = TILE_SIZE * stride + kernelWidth  - 1;
    int tileHeight = TILE_SIZE * stride + kernelHeight - 1;

    for (int c = 0; c < inChannels; c++)
    {
        for (int y = localY; y < tileHeight; y += TILE_SIZE)
        {
            for (int x = localX; x < tileWidth; x += TILE_SIZE)
            {
                int globalX = tileStartX + x;
                int globalY = tileStartY + y;

                float value = 0.0f;

                if (globalX < inWidth && globalY < inHeight)
                {
                    int idx = ((b * inChannels + c) * inHeight + globalY) * inWidth + globalX;
                    value = input[idx];
                }

                Tile[y][x] = value;
            }
        }

        GroupMemoryBarrierWithGroupSync();

        if (outX < outWidth && outY < outHeight)
        {
            int inTileX = localX * stride;
            int inTileY = localY * stride;

            for (int ky = 0; ky < kernelHeight; ky++)
            {
                for (int kx = 0; kx < kernelWidth; kx++)
                {
                    int kernelIdx = ((f * inChannels + c) * kernelHeight + ky) * kernelWidth + kx;

                    sum += Tile[inTileY + ky][inTileX + kx] * kernel[kernelIdx];
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (outX < outWidth && outY < outHeight)
    {
        int outputIdx = ((b * numFilters + f) * outHeight + outY) * outWidth + outX;
        output[outputIdx] = sum;
    }
}


[shader("compute")]
[numthreads(16, 16, 1)]
void conv2d_nchw(
    uint3 id : SV_DispatchThreadID,
    uniform StructuredBuffer<float> input,
    uniform StructuredBuffer<float> kernel,
    uniform RWStructuredBuffer<float> output,
    // input info
    uniform int batchSize,
    uniform int inChannels,
    uniform int inHeight,
    uniform int inWidth,
    // kernel info
    uniform int numFilters,
    uniform int kernelHeight,
    uniform int kernelWidth,
    // conv/out info
    uniform int stride,
    uniform int outHeight,
    uniform int outWidth,
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

    int baseInputBatch = b * inChannels;
    int baseKernelFilter = f * inChannels;

    int inYBase = outY * stride;
    int inXBase = outX * stride;

    for (int c = 0; c < inChannels; c++)
    {
        int inputChannelOffset = (baseInputBatch + c) * inHeight;
        int kernelChannelOffset = (baseKernelFilter + c) * kernelHeight;

        for (int ky = 0; ky < kernelHeight; ky++)
        {
            int iy = inYBase + ky;
            int inputRow = (inputChannelOffset + iy) * inWidth;
            int kernelRow = (kernelChannelOffset + ky) * kernelWidth;

            for (int kx = 0; kx < kernelWidth; kx++)
            {
                int ix = inXBase + kx;
                sum += input[inputRow + ix] * kernel[kernelRow + kx];
            }
        }
    }

    int outputIdx = ((b * numFilters + f) * outHeight + outY) * outWidth + outX;

    output[outputIdx] = sum;
}

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

    for (int f = 0; f < numFilters; f++)
    {
        for (int ky = 0; ky < kernelHeight; ky++)
        {
            int oyNumerator = inY - ky;
            if (oyNumerator < 0 || (oyNumerator % stride) != 0) continue;

            int oy = oyNumerator / stride;
            if (oy >= outHeight) continue;

            for (int kx = 0; kx < kernelWidth; kx++)
            {
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

    int gradInputIdx = ((b * inChannels + c) * inHeight + inY) * inWidth + inX;

    gradInput[gradInputIdx] = sum;
}

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

    for (int b = 0; b < batchSize; b++)
    {
        for (int oy = 0; oy < outHeight; oy++)
        {
            int inY = oy * stride + ky;

            for (int ox = 0; ox < outWidth; ox++)
            {
                int inX = ox * stride + kx;

                int inputIdx = ((b * inChannels + c) * inHeight + inY) * inWidth + inX;
                int gradOutIdx = ((b * numFilters + f) * outHeight + oy) * outWidth + ox;

                sum += input[inputIdx] * gradOutput[gradOutIdx];
            }
        }
    }

    int gradFilterIdx = ((f * inChannels + c) * kernelHeight + ky) * kernelWidth + kx;

    gradFilter[gradFilterIdx] = sum;
}
