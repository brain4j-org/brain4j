static const int TILE_SIZE = 16;
static const int MAX_RANK = 16;

groupshared float Asub[TILE_SIZE][TILE_SIZE];
groupshared float Bsub[TILE_SIZE][TILE_SIZE];

[shader("compute")]
[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void matmul(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uint3 groupThreadID : SV_GroupThreadID,
    uniform StructuredBuffer<float> A,
    uniform StructuredBuffer<float> B,
    uniform RWStructuredBuffer<float> C,
    uniform StructuredBuffer<int> stridesA,
    uniform StructuredBuffer<int> stridesB,
    uniform int M,
    uniform int N,
    uniform int P
)
{
    int row = dispatchThreadID.x;
    int col = dispatchThreadID.y;
    int localRow = groupThreadID.x;
    int localCol = groupThreadID.y;

    int aRowStride = stridesA[0];
    int aColStride = stridesA[1];
    int bRowStride = stridesB[0];
    int bColStride = stridesB[1];

    float sum = 0.0f;
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE_SIZE + localCol;
        int bRow = t * TILE_SIZE + localRow;

        Asub[localRow][localCol] = (row < M && aCol < N) ? A[row * aRowStride + aCol * aColStride] : 0.0f;
        Bsub[localRow][localCol] = (bRow < N && col < P) ? B[bRow * bRowStride + col * bColStride] : 0.0f;

        GroupMemoryBarrierWithGroupSync();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += Asub[localRow][k] * Bsub[k][localCol];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (row < M && col < P) {
        C[row * P + col] = sum;
    }
}
