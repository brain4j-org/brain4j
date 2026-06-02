[shader("compute")]
void add_op(
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
void sub_op(
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
void mul_op(
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
void div_op(
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
void pow_op(
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