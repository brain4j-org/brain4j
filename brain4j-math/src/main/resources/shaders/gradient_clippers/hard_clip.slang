// hard clipping: clamp values to [-bound, bound]

/*
L2 norm clipping: scale down if L2 norm exceeds threshold.
The scale should be computed as min(1, max_norm / (current_norm + epsilon)).
*/

static const int REDUCTION_BLOCK_SIZE = 256;
groupshared float sharedData[REDUCTION_BLOCK_SIZE];

[shader("compute")]
[numthreads(256, 1, 1)]
void hard_clip(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> data,
    uniform float bound,
    uniform int length
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    data[gid] = max(-bound, min(bound, data[gid]));
}
