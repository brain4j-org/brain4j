// hard clipping: clamp values to [-bound, bound]

/*
L2 norm clipping: scale down if L2 norm exceeds threshold.
The scale should be computed as min(1, max_norm / (current_norm + epsilon)).
*/

static const int REDUCTION_BLOCK_SIZE = 256;
groupshared float sharedData[REDUCTION_BLOCK_SIZE];

[shader("compute")]
[numthreads(1, 1, 1)]
void l2_norm_sqrt(
    uniform RWStructuredBuffer<float> normSquared
)
{
    normSquared[0] = sqrt(normSquared[0]);
}
