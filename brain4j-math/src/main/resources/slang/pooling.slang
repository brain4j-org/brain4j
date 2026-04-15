[shader("compute")]
[numthreads(16,16,1)]
void max_pool_forward(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> input,
    uniform RWStructuredBuffer<float> output,
    uniform RWStructuredBuffer<int> xCoords,
    uniform RWStructuredBuffer<int> yCoords,
    uniform int outerSize,
    uniform int inHeight,
    uniform int inWidth,
    uniform int windowHeight,
    uniform int windowWidth,
    uniform int stride,
    uniform int outHeight,
    uniform int outWidth
)
{
    int outX = dispatchThreadID.x;
    int outY = dispatchThreadID.y;
    int outer = dispatchThreadID.z;

    if (outX >= outWidth || outY >= outHeight || outer >= outerSize) return;

    int offsetIn = outer * inHeight * inWidth;
    int offsetOut = outer * outHeight * outWidth;

    int inBaseY = outY * stride;
    int inBaseX = outX * stride;

    float maxVal = -3.402823e38; // -FLT_MAX
    int maxAbs = 0;

    for (int ky = 0; ky < windowHeight; ky++) {
        int iy = inBaseY + ky;
        for (int kx = 0; kx < windowWidth; kx++) {
            int ix = inBaseX + kx;
            if (ix < 0 || ix >= inWidth || iy < 0 || iy >= inHeight) continue;
            int abs = offsetIn + iy * inWidth + ix;
            float v = input[abs];
            if (v > maxVal) {
                maxVal = v;
                maxAbs = abs;
            }
        }
    }

    int outIdx = offsetOut + outY * outWidth + outX;
    output[outIdx] = maxVal;
    int rel = maxAbs - offsetIn;
    yCoords[outIdx] = rel / inWidth;
    xCoords[outIdx] = rel % inWidth;
}


[shader("compute")]
[numthreads(16,16,1)]
void max_pool_backward(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> gradOutput,
    uniform RWStructuredBuffer<float> gradInput,
    uniform StructuredBuffer<int> xCoords,
    uniform StructuredBuffer<int> yCoords,
    uniform int outerSize,
    uniform int inHeight,
    uniform int inWidth,
    uniform int outHeight,
    uniform int outWidth
)
{
    int outX = dispatchThreadID.x;
    int outY = dispatchThreadID.y;
    int outer = dispatchThreadID.z;

    if (outX >= outWidth || outY >= outHeight || outer >= outerSize) return;

    int offsetIn = outer * inHeight * inWidth;
    int offsetOut = outer * outHeight * outWidth;

    int outIdx = offsetOut + outY * outWidth + outX;
    float g = gradOutput[outIdx];

    int y = yCoords[outIdx];
    int x = xCoords[outIdx];
    int inIdx = offsetIn + y * inWidth + x;

    // Atomic add to support overlapping windows
    InterlockedAdd(gradInput[inIdx], g);
}
