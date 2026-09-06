[shader("compute")]
void gather_forward(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> ids,
    uniform StructuredBuffer<float> table,
    uniform RWStructuredBuffer<float> out,
    uniform int totalRows,
    uniform int vocabSize,
    uniform int embedDim
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= totalRows * embedDim) return;

    int row = gid / embedDim;
    int e = gid % embedDim;

    int tid = int(ids[row]);
    tid = tid < 0 ? 0 : (tid >= vocabSize ? vocabSize - 1 : tid);

    out[gid] = table[tid * embedDim + e];
}
