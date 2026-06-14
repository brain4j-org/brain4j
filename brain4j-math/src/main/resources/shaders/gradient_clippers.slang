// hard clipping: clamp values to [-bound, bound]
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

/*
L2 norm clipping: scale down if L2 norm exceeds threshold.
The scale should be computed as min(1, max_norm / (current_norm + epsilon)).
*/
[shader("compute")]
[numthreads(256, 1, 1)]
void l2_clip(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> data,
    uniform float scale,
    uniform int length
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    if (scale < 1.0f) {
        data[gid] *= scale;
    }
}

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

[shader("compute")]
[numthreads(REDUCTION_BLOCK_SIZE, 1, 1)]
void l2_sum_reduce(
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

    sharedData[tid] = gid < length ? data[gid] : 0.0f;

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

[shader("compute")]
[numthreads(REDUCTION_BLOCK_SIZE, 1, 1)]
void l2_norm_squared_final(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uint3 groupThreadID : SV_GroupThreadID,
    uniform RWStructuredBuffer<float> partialSums,
    uniform int numPartialSums
)
{
    uint tid = groupThreadID.x;

    float val = 0.0f;
    if (tid < numPartialSums) {
        val = partialSums[tid];
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
        partialSums[0] = sharedData[0];
    }
}

[shader("compute")]
[numthreads(REDUCTION_BLOCK_SIZE, 1, 1)]
void l2_norm_squared_single_block(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uint3 groupThreadID : SV_GroupThreadID,
    uniform StructuredBuffer<float> data,
    uniform RWStructuredBuffer<float> result,
    uniform int length
)
{
    uint tid = groupThreadID.x;

    float val = 0.0f;
    if (tid < length) {
        val = data[tid];
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
        result[0] = sharedData[0];
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void l2_norm_sqrt(
    uniform RWStructuredBuffer<float> normSquared
)
{
    normSquared[0] = sqrt(normSquared[0]);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void compute_l2_clip_scale(
    uniform RWStructuredBuffer<float> normSquaredBuffer,
    uniform RWStructuredBuffer<float> scaleBuffer,
    uniform float maxNorm,
    uniform float epsilon
)
{
    float normSquared = normSquaredBuffer[0];
    float norm = sqrt(normSquared);
    float scale = maxNorm / (norm + epsilon);
    scaleBuffer[0] = min(1.0f, scale);
}
