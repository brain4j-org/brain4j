// hard clipping: clamp values to [-bound, bound]

/*
L2 norm clipping: scale down if L2 norm exceeds threshold.
The scale should be computed as min(1, max_norm / (current_norm + epsilon)).
*/

static const int REDUCTION_BLOCK_SIZE = 256;
groupshared float sharedData[REDUCTION_BLOCK_SIZE];

[shader("compute")]
[numthreads(256, 1, 1)]
void l2_clip_from_scale(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> data,
    uniform StructuredBuffer<float> scaleBuffer,
    uniform int length
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    float scale = scaleBuffer[0];
    if (scale < 1.0f) {
        data[gid] *= scale;
    }
}
