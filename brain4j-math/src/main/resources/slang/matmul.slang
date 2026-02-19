static const int TILE_SIZE = 16;

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
    uniform int M,
    uniform int N,
    uniform int P,
    uniform int transA,
    uniform int transB
)
{
    int tiled_row = dispatchThreadID.x;
    int tiled_col = dispatchThreadID.y;
    int local_row = groupThreadID.x;
    int local_col = groupThreadID.y;

    float sum = 0.0f;
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        int tiled_col_A = t * TILE_SIZE + local_col;
        int tiled_row_B = t * TILE_SIZE + local_row;

        // "A" tile loading
        if (transA == 0) {
            if (tiled_row < M && tiled_col_A < N)
                Asub[local_row][local_col] = A[tiled_row * N + tiled_col_A];
            else
                Asub[local_row][local_col] = 0.0f;
        } else {
            if (tiled_col_A < N && tiled_row < M)
                Asub[local_row][local_col] = A[tiled_col_A * M + tiled_row];
            else
                Asub[local_row][local_col] = 0.0f;
        }

        // "B" tile loading
        if (transB == 0) {
            if (tiled_row_B < N && tiled_col < P)
                Bsub[local_row][local_col] = B[tiled_row_B * P + tiled_col];
            else
                Bsub[local_row][local_col] = 0.0f;
        } else {
            if (tiled_col < P && tiled_row_B < N)
                Bsub[local_row][local_col] = B[tiled_col * N + tiled_row_B];
            else
                Bsub[local_row][local_col] = 0.0f;
        }

        GroupMemoryBarrierWithGroupSync();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += Asub[local_row][k] * Bsub[k][local_col];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (tiled_row < M && tiled_col < P)
        C[tiled_row * P + tiled_col] = sum;
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

    int local_row = groupThreadID.x;
    int local_col = groupThreadID.y;

    if (batch >= batchCount) return;

    int A_offset = offsetsA[batch];
    int B_offset = offsetsB[batch];
    int C_offset = offsetsC[batch];

    int K = N;

    int A_rowStride = (transA == 0) ? K : M;
    int B_rowStride = (transB == 0) ? P : K;

    float sum = 0.0f;

    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        int tiled_k = t * TILE_SIZE;

        int a_r = row;
        int a_c = tiled_k + local_col;
        int b_r = tiled_k + local_row;
        int b_c = col;

        // "A" tile loading
        if (a_r < M && a_c < K) {
            if (transA == 0) {
                Asub[local_row][local_col] = A[A_offset + a_r * A_rowStride + a_c];
            } else {
                Asub[local_row][local_col] = A[A_offset + a_c * A_rowStride + a_r];
            }
        } else {
            Asub[local_row][local_col] = 0.0f;
        }

        // "B" tile loading
        if (b_r < K && b_c < P) {
            if (transB == 0) {
                Bsub[local_row][local_col] = B[B_offset + b_r * B_rowStride + b_c];
            } else {
                Bsub[local_row][local_col] = B[B_offset + b_c * B_rowStride + b_r];
            }
        } else {
            Bsub[local_row][local_col] = 0.0f;
        }

        GroupMemoryBarrierWithGroupSync();

        // partial sum computing
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += Asub[local_row][k] * Bsub[k][local_col];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (row < M && col < P) {
        C[C_offset + row * P + col] = sum;
    }
}
