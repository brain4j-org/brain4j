static const int TILE_SIZE = 16;
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
void matmul_naive(
    uint3 id : SV_DispatchThreadID,

    StructuredBuffer<float> A,
    StructuredBuffer<float> B,
    RWStructuredBuffer<float> C,

    StructuredBuffer<int> offsetsA,
    StructuredBuffer<int> offsetsB,
    StructuredBuffer<int> offsetsC,

    uniform int M,
    uniform int K,
    uniform int P,
    uniform int batchCount
)
{
    uint row = id.x;
    uint col = id.y;
    uint batch = id.z;

    if (batch >= batchCount) return;
    if (row >= M || col >= P) return;

    int Aoff = offsetsA[batch];
    int Boff = offsetsB[batch];
    int Coff = offsetsC[batch];

    float sum = 0.0f;

    for (uint k = 0; k < K; ++k) {
        float a = A[Aoff + row * K + k];
        float b = B[Boff + k * P + col];
        sum += a * b;
    }

    C[Coff + row * P + col] = sum;
}

groupshared float Asub[TILE_SIZE][TILE_SIZE];
groupshared float Bsub[TILE_SIZE][TILE_SIZE];

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

    int K = (transA == 0) ? N : M;

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

[shader("compute")]
[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void matmul_legacy(
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
    uniform int batchCount
)
{
    int row = dispatchThreadID.x;
    int col = dispatchThreadID.y;
    int batch = dispatchThreadID.z;

    int local_row = groupThreadID.x;
    int local_col = groupThreadID.y;

    if (batch >= batchCount) return;

    bool valid = (row < M) && (col < P);

    int A_offset = offsetsA[batch];
    int B_offset = offsetsB[batch];
    int C_offset = offsetsC[batch];

    float sum = 0.0f;
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        int tiled_col_A = t * TILE_SIZE + local_col;
        int tiled_row_B = t * TILE_SIZE + local_row;

        // "A" tile loading
        if (row < M && tiled_col_A < N) {
            Asub[local_row][local_col] = A[A_offset + row * N + tiled_col_A];
        } else {
            Asub[local_row][local_col] = 0.0f;
        }

        // "B" tile loading
        if (tiled_row_B < N && col < P) {
            Bsub[local_row][local_col] = B[B_offset + tiled_row_B * P + col];
        } else {
            Bsub[local_row][local_col] = 0.0f;
        }

        GroupMemoryBarrierWithGroupSync();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += Asub[local_row][k] * Bsub[k][local_col];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (valid) {
        C[C_offset + row * P + col] = sum;
    }
}

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
            if (tiled_col_A < M && tiled_row < N)
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
            if (tiled_col < N && tiled_row_B < P)
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
void add(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> a,
    uniform StructuredBuffer<float> b,
    uniform int size,
    uniform int broadcast_dim,
    uniform int batch
)
{
    uint gid = dispatchThreadID.x;

    if (gid >= size) return;

    if (broadcast_dim == -1) {
        a[gid] += b[gid];
    } else {
        int j = gid % broadcast_dim;
        a[gid] += b[j];
    }
}

[shader("compute")]
void sub(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> a,
    uniform StructuredBuffer<float> b,
    uniform int size,
    uniform int broadcast_dim,
    uniform int batch
)
{
    uint gid = dispatchThreadID.x;

    if (broadcast_dim == -1) {
        if (gid < size) {
            a[gid] -= b[gid];
        }
    } else {
        if (gid < size) {
            int j = gid % broadcast_dim;
            a[gid] -= b[j];
        }
    }
}

[shader("compute")]
void mul(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> a,
    uniform StructuredBuffer<float> b,
    uniform int size,
    uniform int broadcast_dim,
    uniform int batch
)
{
    uint gid = dispatchThreadID.x;

    if (broadcast_dim == -1) {
        if (gid < size) {
            a[gid] *= b[gid];
        }
    } else {
        if (gid < size) {
            int j = gid % broadcast_dim;
            a[gid] *= b[j];
        }
    }
}

[shader("compute")]
void div_op(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> a,
    uniform StructuredBuffer<float> b,
    uniform int size,
    uniform int broadcast_dim,
    uniform int batch
)
{
    uint gid = dispatchThreadID.x;

    if (broadcast_dim == -1) {
        if (gid < size) {
            a[gid] /= b[gid];
        }
    } else {
        if (gid < size) {
            int j = gid % broadcast_dim;
            a[gid] /= b[j];
        }
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
    uniform int lastDim,
    uniform float temperature
)
{
    int row = dispatchThreadID.x;
    int offset = row * lastDim;

    // look for max for numerical stability
    float max_val = input[offset];
    for (int i = 1; i < lastDim; i++) {
        float val = input[offset + i];
        if (val > max_val) max_val = val;
    }

    // compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < lastDim; i++) {
        float val = (input[offset + i] - max_val) / temperature;
        float exp_val = exp(val);
        output[offset + i] = exp_val;
        sum += exp_val;
    }

    // normalization
    for (int i = 0; i < lastDim; i++) {
        output[offset + i] /= sum;
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
        int aIdx = row * lastA + col;
        C[gid] = A[aIdx];
    } else {
        int bIdx = row * lastB + (col - lastA);
        C[gid] = B[bIdx];
    }
}

[shader("compute")]
void concat_copy_a(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> A,
    uniform RWStructuredBuffer<float> C,
    uniform int numBlocks,
    uniform int thisDim,
    uniform int otherDim,
    uniform int blockSize
)
{
    int gid = dispatchThreadID.x;
    int blockElem = thisDim * blockSize;
    int total = numBlocks * blockElem;
    if (gid >= total) return;

    int block = gid / blockElem;
    int inBlockIdx = gid % blockElem;

    int destStride = (thisDim + otherDim) * blockSize;
    int destIndex = block * destStride + inBlockIdx;

    C[destIndex] = A[gid];
}

[shader("compute")]
void concat_copy_b(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> B,
    uniform RWStructuredBuffer<float> C,
    uniform int numBlocks,
    uniform int thisDim,
    uniform int otherDim,
    uniform int blockSize
)
{
    int gid = dispatchThreadID.x;
    int blockElem = otherDim * blockSize;
    int total = numBlocks * blockElem;
    if (gid >= total) return;

    int block = gid / blockElem;
    int inBlockIdx = gid % blockElem;

    int destStride = (thisDim + otherDim) * blockSize;
    int destIndex = block * destStride + thisDim * blockSize + inBlockIdx;

    C[destIndex] = B[gid];
}

