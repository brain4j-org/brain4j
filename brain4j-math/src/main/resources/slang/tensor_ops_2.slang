[shader("compute")]
void add(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> a,
    uniform StructuredBuffer<float> b,
    uniform int size,
    uniform StructuredBuffer<int> aShape,
    uniform StructuredBuffer<int> aFlatStrides,
    uniform StructuredBuffer<int> bEffStrides,
    uniform int rank,
    uniform int sameShape
)
{
    uint gid = dispatchThreadID.x;

    if (gid >= size) return;

    if (sameShape == 1) {
        a[gid] += b[gid];
    } else {
        int bIdx = 0;
        for (int d = 0; d < rank; d++) {
            int idx = (gid / aFlatStrides[d]) % aShape[d];
            bIdx += idx * bEffStrides[d];
        }
        a[gid] += b[bIdx];
    }
}

[shader("compute")]
void sub(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> a,
    uniform StructuredBuffer<float> b,
    uniform int size,
    uniform StructuredBuffer<int> aShape,
    uniform StructuredBuffer<int> aFlatStrides,
    uniform StructuredBuffer<int> bEffStrides,
    uniform int rank,
    uniform int sameShape
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= size) return;

    if (sameShape == 1) {
        a[gid] -= b[gid];
    } else {
        int bIdx = 0;
        for (int d = 0; d < rank; d++) {
            int idx = (gid / aFlatStrides[d]) % aShape[d];
            bIdx += idx * bEffStrides[d];
        }
        a[gid] -= b[bIdx];
    }
}

[shader("compute")]
void mul(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> a,
    uniform StructuredBuffer<float> b,
    uniform int size,
    uniform StructuredBuffer<int> aShape,
    uniform StructuredBuffer<int> aFlatStrides,
    uniform StructuredBuffer<int> bEffStrides,
    uniform int rank,
    uniform int sameShape
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= size) return;

    if (sameShape == 1) {
        a[gid] *= b[gid];
    } else {
        int bIdx = 0;
        for (int d = 0; d < rank; d++) {
            int idx = (gid / aFlatStrides[d]) % aShape[d];
            bIdx += idx * bEffStrides[d];
        }
        a[gid] *= b[bIdx];
    }
}

[shader("compute")]
void div(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> a,
    uniform StructuredBuffer<float> b,
    uniform int size,
    uniform StructuredBuffer<int> aShape,
    uniform StructuredBuffer<int> aFlatStrides,
    uniform StructuredBuffer<int> bEffStrides,
    uniform int rank,
    uniform int sameShape
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= size) return;

    if (sameShape == 1) {
        a[gid] /= b[gid];
    } else {
        int bIdx = 0;
        for (int d = 0; d < rank; d++) {
            int idx = (gid / aFlatStrides[d]) % aShape[d];
            bIdx += idx * bEffStrides[d];
        }
        a[gid] /= b[bIdx];
    }
}

[shader("compute")]
void pow(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> a,
    uniform StructuredBuffer<float> b,
    uniform int size,
    uniform StructuredBuffer<int> aShape,
    uniform StructuredBuffer<int> aFlatStrides,
    uniform StructuredBuffer<int> bEffStrides,
    uniform int rank,
    uniform int sameShape
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= size) return;

    if (sameShape == 1) {
        a[gid] = pow(a[gid], b[gid]);
    } else {
        int bIdx = 0;
        for (int d = 0; d < rank; d++) {
            int idx = (gid / aFlatStrides[d]) % aShape[d];
            bIdx += idx * bEffStrides[d];
        }
        a[gid] = pow(a[gid], b[bIdx]);
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void sum_along_dim(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> input,
    uniform RWStructuredBuffer<float> output,
    uniform int outerSize,
    uniform int reducedSize,
    uniform int innerSize
)
{
    int gid_outer = dispatchThreadID.x;
    int gid_inner = dispatchThreadID.y;

    if (gid_outer >= outerSize || gid_inner >= innerSize) return;

    float sum = 0.0f;
    for (int i = 0; i < reducedSize; i++) {
        int idx = gid_outer * reducedSize * innerSize + i * innerSize + gid_inner;
        sum += input[idx];
    }

    int resultIndex = gid_outer * innerSize + gid_inner;
    output[resultIndex] = sum;
}

[shader("compute")]
void softmax_last_dim(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> input,
    uniform RWStructuredBuffer<float> output,
    uniform StructuredBuffer<int> rowOffsets,
    uniform int lastStride,
    uniform int rows,
    uniform int lastDim,
    uniform float temperature
)
{
    int row = dispatchThreadID.x;
    if (row >= rows) return;

    int inputBaseOffset = rowOffsets[row];
    int outputBaseOffset = row * lastDim;

    // look for max for numerical stability
    float max_val = input[inputBaseOffset];
    for (int i = 1; i < lastDim; i++) {
        float val = input[inputBaseOffset + i * lastStride];
        if (val > max_val) max_val = val;
    }

    // compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < lastDim; i++) {
        float val = (input[inputBaseOffset + i * lastStride] - max_val) / temperature;
        float exp_val = exp(val);
        output[outputBaseOffset + i] = exp_val;
        sum += exp_val;
    }

    // normalization
    for (int i = 0; i < lastDim; i++) {
        output[outputBaseOffset + i] /= sum;
    }
}
