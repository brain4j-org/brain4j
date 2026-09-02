// hard clipping: clamp values to [-bound, bound]

/*
L2 norm clipping: scale down if L2 norm exceeds threshold.
The scale should be computed as min(1, max_norm / (current_norm + epsilon)).
*/

static const int REDUCTION_BLOCK_SIZE = 256;
groupshared float sharedData[REDUCTION_BLOCK_SIZE];

[shader("compute")]
[numthreads(REDUCTION_BLOCK_SIZE, 1, 1)]
void l2_norm_squared_reduce(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uint3 groupThreadID : SV_GroupThreadID,
    uint3 groupID : SV_GroupID,
    uniform StructuredBuffer<float> data,
    uniform RWStructuredBuffer<float> partialSums,
    uniform int length
)
{
    uint tid = groupThreadID.x;
    uint gid = dispatchThreadID.x;
    uint blockIdx = groupID.x;

    float val = 0.0f;
    if (gid < length) {
        val = data[gid];
        val = val * val;
    }
    sharedData[tid] = val;

    GroupMemoryBarrierWithGroupSync();

    for (uint stride = REDUCTION_BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0) {
        partialSums[blockIdx] = sharedData[0];
    }
}
