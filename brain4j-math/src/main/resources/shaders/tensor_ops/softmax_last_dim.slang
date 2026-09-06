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
