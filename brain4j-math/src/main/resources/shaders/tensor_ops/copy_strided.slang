// Strided copy: dst[linear] = src[sum(coord_d * strides[d])] for every linear
// index of the destination shape. Handles arbitrary views (e.g. transposed)
// without any host-side traffic.
[shader("compute")]
void copy_strided(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> src,
    uniform RWStructuredBuffer<float> dst,
    uniform StructuredBuffer<int> strides,
    uniform StructuredBuffer<int> shape,
    uniform int rank,
    uniform int length
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    int offset = 0;
    int tmp = gid;
    for (int d = rank - 1; d >= 0; d--) {
        int coord = tmp % shape[d];
        tmp /= shape[d];
        offset += coord * strides[d];
    }

    dst[gid] = src[offset];
}
