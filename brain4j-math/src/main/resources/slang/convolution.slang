[shader("compute")]
[numthreads(16, 16, 1)]
void conv2d_nchw_shared(
    uint3 dispatchThreadID: SV_DispatchThreadID,
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
    uniform int outWidth
)
{
     // TODO
}

[shader("compute")]
[numthreads(16, 16, 1)]
void conv2d_nchw(
    uint3 dispatchThreadID : SV_DispatchThreadID,
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
    uniform int outWidth
)
{
    int outX = dispatchThreadID.x;
    int outY = dispatchThreadID.y;
    int bf = dispatchThreadID.z;

    if (outX >= outWidth || outY >= outHeight) return;
    if (bf >= batchSize * numFilters) return;

    int b = bf / numFilters;
    int f = bf % numFilters;

    int batchStride = b * inChannels;
    int filterStride = f * inChannels;

    float sum = 0.0f;
    int outYStride = outY * stride;
    int outXStride = outX * stride;

    for (int c = 0; c < inChannels; c++) {
        int channelBStride = (batchStride + c) * inHeight;
        int channelFStride = (filterStride + c) * kernelHeight;

        for (int ky = 0; ky < kernelHeight; ky++) {
            int iy = outYStride + ky;
            int inputStride = (channelBStride + iy) * inWidth;
            int kernelStride = (channelFStride + ky) * kernelWidth;

            for (int kx = 0; kx < kernelWidth; kx++) {
                int ix = outXStride + kx;

                int inputIdx = inputStride + ix;
                int kernelIdx = kernelStride + kx;

                sum += input[inputIdx] * kernel[kernelIdx];
            }
        }
    }

    int outputIdx = ((b * numFilters + f) * outHeight + outY) * outWidth + outX;
    output[outputIdx] = sum;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void conv2d_backward_input_nchw(
    uint3 dispatchThreadID : SV_DispatchThreadID,
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
    int inX = dispatchThreadID.x;
    int inY = dispatchThreadID.y;
    int bc = dispatchThreadID.z;

    if (inX >= inWidth || inY >= inHeight) return;
    if (bc >= batchSize * inChannels) return;

    int b = bc / inChannels;
    int c = bc % inChannels;

    float sum = 0.0f;

    for (int f = 0; f < numFilters; f++) {
        for (int ky = 0; ky < kernelHeight; ky++) {
            int oyNumerator = inY - ky;
            if (oyNumerator < 0 || (oyNumerator % stride) != 0) continue;

            int oy = oyNumerator / stride;
            if (oy < 0 || oy >= outHeight) continue;

            for (int kx = 0; kx < kernelWidth; kx++) {
                int oxNumerator = inX - kx;
                if (oxNumerator < 0 || (oxNumerator % stride) != 0) continue;

                int ox = oxNumerator / stride;
                if (ox < 0 || ox >= outWidth) continue;

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
    uint3 dispatchThreadID : SV_DispatchThreadID,
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
    int kx = dispatchThreadID.x;
    int ky = dispatchThreadID.y;
    int fc = dispatchThreadID.z;

    if (kx >= kernelWidth || ky >= kernelHeight) return;
    if (fc >= numFilters * inChannels) return;

    int f = fc / inChannels;
    int c = fc % inChannels;

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

    int gradFilterIdx = ((f * inChannels + c) * kernelHeight + ky) * kernelWidth + kx;
    gradFilter[gradFilterIdx] = sum;
}
