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
void concat_last_dim(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> A,
    uniform StructuredBuffer<float> B,
    uniform RWStructuredBuffer<float> C,
    uniform int outerSize,
    uniform int lastA,
    uniform int lastB,
    uniform int concatLast
)
{
    int gid = dispatchThreadID.x;
    int total = outerSize * concatLast;
    if (gid >= total) return;

    int row = gid / concatLast;
    int col = gid % concatLast;

    if (col < lastA) {
        C[gid] = A[row * lastA + col];
    } else {
        C[gid] = B[row * lastB + (col - lastA)];
    }
}
