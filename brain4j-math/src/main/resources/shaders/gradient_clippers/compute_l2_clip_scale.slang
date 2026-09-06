// hard clipping: clamp values to [-bound, bound]

/*
L2 norm clipping: scale down if L2 norm exceeds threshold.
The scale should be computed as min(1, max_norm / (current_norm + epsilon)).
*/

static const int REDUCTION_BLOCK_SIZE = 256;
groupshared float sharedData[REDUCTION_BLOCK_SIZE];

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
