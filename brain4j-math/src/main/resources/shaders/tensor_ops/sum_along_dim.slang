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
    int gidOuter = dispatchThreadID.x;
    int gidInner = dispatchThreadID.y;
    if (gidOuter >= outerSize || gidInner >= innerSize) return;

    float sum = 0.0f;
    for (int i = 0; i < reducedSize; i++) {
        int idx = gidOuter * reducedSize * innerSize + i * innerSize + gidInner;
        sum += input[idx];
    }

    output[gidOuter * innerSize + gidInner] = sum;
}
