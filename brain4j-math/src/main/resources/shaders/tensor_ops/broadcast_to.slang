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
