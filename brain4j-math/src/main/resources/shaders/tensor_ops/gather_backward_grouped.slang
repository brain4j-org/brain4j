[shader("compute")]
void gather_backward_grouped(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> gradOut,
    uniform StructuredBuffer<int> uniqueIds,
    uniform StructuredBuffer<int> rowPtr,
    uniform StructuredBuffer<int> positions,
    uniform RWStructuredBuffer<float> gradTable,
    uniform int numUnique,
    uniform int embedDim
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= numUnique) return;

    int uid = uniqueIds[gid];
    int base = uid * embedDim;

    for (int k = rowPtr[gid]; k < rowPtr[gid + 1]; k++) {
        int rowBase = positions[k] * embedDim;

        for (int e = 0; e < embedDim; e++) {
            gradTable[base + e] += gradOut[rowBase + e];
        }
    }
}
