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
[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void matmul_batched(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uint3 groupThreadID : SV_GroupThreadID,
    uint3 groupID : SV_GroupID,
    uniform StructuredBuffer<float> A,
    uniform StructuredBuffer<float> B,
    uniform RWStructuredBuffer<float> C,
    uniform StructuredBuffer<int> offsetsA,
    uniform StructuredBuffer<int> offsetsB,
    uniform StructuredBuffer<int> offsetsC,
    uniform int M,
    uniform int N,
    uniform int P,
    uniform int batchCount,
    uniform int transA,
    uniform int transB
)
{
    int row = dispatchThreadID.x;
    int col = dispatchThreadID.y;
    int batch = dispatchThreadID.z;
    int localRow = groupThreadID.x;
    int localCol = groupThreadID.y;

    if (batch >= batchCount) return;

    int aOffset = offsetsA[batch];
    int bOffset = offsetsB[batch];
    int cOffset = offsetsC[batch];
    int kSize = N;
    int aRowStride = (transA == 0) ? kSize : M;
    int bRowStride = (transB == 0) ? P : kSize;

    float sum = 0.0f;
    int numTiles = (kSize + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        int tiledK = t * TILE_SIZE;
        int aCol = tiledK + localCol;
        int bRow = tiledK + localRow;

        if (row < M && aCol < kSize) {
            Asub[localRow][localCol] = transA == 0
                ? A[aOffset + row * aRowStride + aCol]
                : A[aOffset + aCol * aRowStride + row];
        } else {
            Asub[localRow][localCol] = 0.0f;
        }

        if (bRow < kSize && col < P) {
            Bsub[localRow][localCol] = transB == 0
                ? B[bOffset + bRow * bRowStride + col]
                : B[bOffset + col * bRowStride + bRow];
        } else {
            Bsub[localRow][localCol] = 0.0f;
        }

        GroupMemoryBarrierWithGroupSync();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += Asub[localRow][k] * Bsub[k][localCol];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (row < M && col < P) {
        C[cOffset + row * P + col] = sum;
    }
}
