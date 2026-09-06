static const int TILE_SIZE = 16;
static const int MAX_RANK = 16;

groupshared float Asub[TILE_SIZE][TILE_SIZE];
groupshared float Bsub[TILE_SIZE][TILE_SIZE];

void legacy_binary_op(uint gid, int op, RWStructuredBuffer<float> a, StructuredBuffer<float> b, int size, int broadcastDim, int batch)
{
    if (gid >= size) return;

    int bIdx = broadcastDim < 0 ? int(gid) : int(gid) % broadcastDim;
    float rhs = b[bIdx];

    if (op == 0) a[gid] += rhs;
    if (op == 1) a[gid] -= rhs;
    if (op == 2) a[gid] *= rhs;
    if (op == 3) a[gid] /= rhs;
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
            int idx = (int(gid) / aFlatStrides[d]) % aShape[d];
            bIdx += idx * bEffStrides[d];
        }
        a[gid] /= b[bIdx];
    }
}
