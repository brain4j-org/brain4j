[shader("compute")]
[numthreads(16, 16, 1)]
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
    int inX = dispatchThreadID.x;
    int inY = dispatchThreadID.y;
    int outer = dispatchThreadID.z;
    if (inX >= inWidth || inY >= inHeight || outer >= outerSize) return;

    int offsetIn = outer * inHeight * inWidth;
    int offsetOut = outer * outHeight * outWidth;
    float sum = 0.0f;

    for (int outY = 0; outY < outHeight; outY++) {
        for (int outX = 0; outX < outWidth; outX++) {
            int outIdx = offsetOut + outY * outWidth + outX;
            if (yCoords[outIdx] == inY && xCoords[outIdx] == inX) {
                sum += gradOutput[outIdx];
            }
        }
    }

    gradInput[offsetIn + inY * inWidth + inX] = sum;
}
