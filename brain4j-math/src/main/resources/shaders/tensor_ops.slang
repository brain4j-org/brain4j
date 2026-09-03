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
    int localRow = groupThreadID.x;
    int localCol = groupThreadID.y;

    if (batch >= batchCount) return;

    int aOffset = offsetsA[batch];
    int bOffset = offsetsB[batch];
    int cOffset = offsetsC[batch];

    float sum = 0.0f;
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE_SIZE + localCol;
        int bRow = t * TILE_SIZE + localRow;

        Asub[localRow][localCol] = (row < M && aCol < N) ? A[aOffset + row * N + aCol] : 0.0f;
        Bsub[localRow][localCol] = (bRow < N && col < P) ? B[bOffset + bRow * P + col] : 0.0f;

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

    int base = int(batchIdx) * features;

    float mean = 0.0f;
    for (int j = 0; j < features; j++) mean += input[base + j];
    mean /= float(features);

    float var = 0.0f;
    for (int j = 0; j < features; j++) {
        float diff = input[base + j] - mean;
        var += diff * diff;
    }
    var /= float(features);

    float denom = sqrt(var + epsilon);
    for (int j = 0; j < features; j++) {
        output[base + j] = (input[base + j] - mean) / denom;
    }
}

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
            int idx = (int(gid) / aFlatStrides[d]) % aShape[d];
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
            int idx = (int(gid) / aFlatStrides[d]) % aShape[d];
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
            int idx = (int(gid) / aFlatStrides[d]) % aShape[d];
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
            int idx = (int(gid) / aFlatStrides[d]) % aShape[d];
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
            int idx = (int(gid) / aFlatStrides[d]) % aShape[d];
            bIdx += idx * bEffStrides[d];
        }
        a[gid] = pow(a[gid], b[bIdx]);
    }
}

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
void add(uint3 id : SV_DispatchThreadID, uniform RWStructuredBuffer<float> a, uniform StructuredBuffer<float> b, uniform int size, uniform int broadcast_dim, uniform int batch)
{
    legacy_binary_op(id.x, 0, a, b, size, broadcast_dim, batch);
}

[shader("compute")]
void sub(uint3 id : SV_DispatchThreadID, uniform RWStructuredBuffer<float> a, uniform StructuredBuffer<float> b, uniform int size, uniform int broadcast_dim, uniform int batch)
{
    legacy_binary_op(id.x, 1, a, b, size, broadcast_dim, batch);
}

[shader("compute")]
void mul(uint3 id : SV_DispatchThreadID, uniform RWStructuredBuffer<float> a, uniform StructuredBuffer<float> b, uniform int size, uniform int broadcast_dim, uniform int batch)
{
    legacy_binary_op(id.x, 2, a, b, size, broadcast_dim, batch);
}

[shader("compute")]
void div(uint3 id : SV_DispatchThreadID, uniform RWStructuredBuffer<float> a, uniform StructuredBuffer<float> b, uniform int size, uniform int broadcast_dim, uniform int batch)
{
    legacy_binary_op(id.x, 3, a, b, size, broadcast_dim, batch);
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

    float maxVal = input[inputBaseOffset];
    for (int i = 1; i < lastDim; i++) {
        float val = input[inputBaseOffset + i * lastStride];
        if (val > maxVal) maxVal = val;
    }

    float sum = 0.0f;
    for (int i = 0; i < lastDim; i++) {
        float val = (input[inputBaseOffset + i * lastStride] - maxVal) / temperature;
        float expVal = exp(val);
        output[outputBaseOffset + i] = expVal;
        sum += expVal;
    }

    for (int i = 0; i < lastDim; i++) {
        output[outputBaseOffset + i] /= sum;
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
void slice_backward_scatter(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> gradOutput,
    uniform RWStructuredBuffer<float> gradInput,
    uniform StructuredBuffer<int> inputStrides,
    uniform StructuredBuffer<int> outputStrides,
    uniform StructuredBuffer<int> outputShape,
    uniform StructuredBuffer<int> starts,
    uniform StructuredBuffer<int> steps,
    uniform int rank
)
{
    int outputLinearIdx = dispatchThreadID.x;

    int totalElements = 1;
    for (int i = 0; i < rank; i++) totalElements *= outputShape[i];
    if (outputLinearIdx >= totalElements) return;

    int tmp = outputLinearIdx;
    int inputOffset = 0;
    int outputOffset = 0;

    for (int i = 0; i < rank; i++) {
        int coord = tmp / outputStrides[i];
        tmp = tmp % outputStrides[i];
        inputOffset += (starts[i] + coord * steps[i]) * inputStrides[i];
        outputOffset += coord * outputStrides[i];
    }

    gradInput[inputOffset] += gradOutput[outputOffset];
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
    C[block * destStride + inBlockIdx] = A[gid];
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
    C[block * destStride + thisDim * blockSize + inBlockIdx] = B[gid];
}

[shader("compute")]
void broadcast_to(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> srcData,
    uniform RWStructuredBuffer<float> dstData,
    uniform StructuredBuffer<int> targetShape,
    uniform StructuredBuffer<int> targetFlatStrides,
    uniform StructuredBuffer<int> alignedSrcShape,
    uniform StructuredBuffer<int> alignedSrcStrides,
    uniform int rank,
    uniform int totalElements
)
{
    int gid = dispatchThreadID.x;
    if (gid >= totalElements) return;

    int srcOffset = 0;
    int useZero = 0;

    for (int d = 0; d < rank; d++) {
        int coord = (gid / targetFlatStrides[d]) % targetShape[d];
        int srcDim = alignedSrcShape[d];
        int srcCoord = 0;

        if (srcDim == targetShape[d]) {
            srcCoord = coord;
        } else if (srcDim == 1) {
            srcCoord = 0;
        } else if (coord < srcDim) {
            srcCoord = coord;
        } else {
            useZero = 1;
            break;
        }

        srcOffset += srcCoord * alignedSrcStrides[d];
    }

    dstData[gid] = useZero == 1 ? 0.0f : srcData[srcOffset];
}

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
