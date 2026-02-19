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
