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

[shader("compute")]
[numthreads(256, 1, 1)]
void convolve1d_direct(
    uint3 id : SV_DispatchThreadID,
    uniform StructuredBuffer<float> input,
    uniform StructuredBuffer<float> kernel,
    uniform RWStructuredBuffer<float> output,
    uniform int inputSize,
    uniform int kernelSize,
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

[shader("compute")]
[numthreads(16, 16, 1)]
void convolve2d_direct(
    uint3 id : SV_DispatchThreadID,
    uniform StructuredBuffer<float> input,
    uniform StructuredBuffer<float> kernel,
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
    int outRow = id.x;
    int outCol = id.y;
    if (outRow >= outputRows || outCol >= outputCols) return;

    float sum = 0.0f;
    for (int kr = 0; kr < kernelRows; kr++) {
        int inRow = outRow + kr - paddingTop;
        if (inRow < 0 || inRow >= inputRows) continue;

        for (int kc = 0; kc < kernelCols; kc++) {
            int inCol = outCol + kc - paddingLeft;
            if (inCol < 0 || inCol >= inputCols) continue;

            sum += input[inRow * inputCols + inCol] * kernel[kr * kernelCols + kc];
        }
    }

    output[outRow * outputCols + outCol] = sum;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void convolve2d_fft_extract(
    uint3 id : SV_DispatchThreadID,
    uniform StructuredBuffer<float> result,
    uniform RWStructuredBuffer<float> output,
    uniform int outputRows,
    uniform int outputCols,
    uniform int startRow,
    uniform int startCol,
    uniform int resultCols
)
{
    int row = id.x;
    int col = id.y;
    if (row >= outputRows || col >= outputCols) return;

    output[row * outputCols + col] = result[(row + startRow) * resultCols + (col + startCol)];
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
