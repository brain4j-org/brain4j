static const int MAX_RANK = 16;

[shader("compute")]
void layer_norm(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> input,
    uniform RWStructuredBuffer<float> output,
    uniform int features,
    uniform int batchSize,
    uniform float epsilon
)
{
    uint batchIdx = dispatchThreadID.x;
    if (batchIdx >= batchSize) return;

    int base = batchIdx * features;

    float mean = 0.0f;
    for (int j = 0; j < features; j++) {
        mean += input[base + j];
    }
    mean /= float(features);

    float var = 0.0f;
    for (int j = 0; j < features; j++) {
        float diff = input[base + j] - mean;
        var += diff * diff;
    }
    var /= float(features);

    // normalization
    float denom = sqrt(var + epsilon);
    for (int j = 0; j < features; j++) {
        output[base + j] = (input[base + j] - mean) / denom;
    }
}

[shader("compute")]
void slice(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> srcData,
    uniform RWStructuredBuffer<float> dstData,
    uniform StructuredBuffer<int> srcStrides,
    uniform StructuredBuffer<int> dstStrides,
    uniform StructuredBuffer<int> dstShape,
    uniform StructuredBuffer<int> starts,
    uniform StructuredBuffer<int> steps,
    uniform int rank
)
{
    int dstLinearIdx = dispatchThreadID.x;

    // compute total elements
    int totalElements = 1;
    for (int i = 0; i < rank; i++) totalElements *= dstShape[i];
    if (dstLinearIdx >= totalElements) return;

    int tmp = dstLinearIdx;
    int srcOffset = 0;

    for (int i = 0; i < rank; i++) {
        int idx = tmp / dstStrides[i];
        tmp = tmp % dstStrides[i];

        int srcIdx = starts[i] + idx * steps[i];
        srcOffset += srcIdx * srcStrides[i];
    }

    dstData[dstLinearIdx] = srcData[srcOffset];
}

[shader("compute")]
void broadcast_to(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> srcData,
    uniform RWStructuredBuffer<float> dstData,
    uniform StructuredBuffer<int> targetShape,
    uniform StructuredBuffer<int> targetFlatStrides,
    uniform StructuredBuffer<int> alignedSrcShape,
    uniform StructuredBuffer<int> alignedSrcStrides,
    uniform int rank,
    uniform int totalElements
)
{
    int gid = dispatchThreadID.x;
    if (gid >= totalElements) return;

    int srcOffset = 0;
    int useZero = 0;

    for (int d = 0; d < rank; d++) {
        int coord = (gid / targetFlatStrides[d]) % targetShape[d];
        int srcDim = alignedSrcShape[d];
        int srcCoord = 0;

        if (srcDim == targetShape[d]) {
            srcCoord = coord;
        } else if (srcDim == 1) {
            srcCoord = 0;
        } else if (coord < srcDim) {
            srcCoord = coord;
        } else {
            useZero = 1;
            break;
        }

        srcOffset += srcCoord * alignedSrcStrides[d];
    }

    dstData[gid] = useZero == 1 ? 0.0f : srcData[srcOffset];
}
