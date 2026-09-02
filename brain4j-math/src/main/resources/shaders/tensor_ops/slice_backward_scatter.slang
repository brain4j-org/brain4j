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
