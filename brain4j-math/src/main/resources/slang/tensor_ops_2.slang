[shader("compute")]
void add(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> a,
    uniform StructuredBuffer<float> b,
    uniform int size,
    uniform int broadcast_dim,
    uniform int batch
)
{
    uint gid = dispatchThreadID.x;

    if (gid >= size) return;

    if (broadcast_dim == -1) {
        a[gid] += b[gid];
    } else {
        int j = gid % broadcast_dim;
        a[gid] += b[j];
    }
}

[shader("compute")]
void sub(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> a,
    uniform StructuredBuffer<float> b,
    uniform int size,
    uniform int broadcast_dim,
    uniform int batch
)
{
    uint gid = dispatchThreadID.x;

    if (broadcast_dim == -1) {
        if (gid < size) {
            a[gid] -= b[gid];
        }
    } else {
        if (gid < size) {
            int j = gid % broadcast_dim;
            a[gid] -= b[j];
        }
    }
}

[shader("compute")]
void mul(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> a,
    uniform StructuredBuffer<float> b,
    uniform int size,
    uniform int broadcast_dim,
    uniform int batch
)
{
    uint gid = dispatchThreadID.x;

    if (broadcast_dim == -1) {
        if (gid < size) {
            a[gid] *= b[gid];
        }
    } else {
        if (gid < size) {
            int j = gid % broadcast_dim;
            a[gid] *= b[j];
        }
    }
}

[shader("compute")]
void div(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> a,
    uniform StructuredBuffer<float> b,
    uniform int size,
    uniform int broadcast_dim,
    uniform int batch
)
{
    uint gid = dispatchThreadID.x;

    if (broadcast_dim == -1) {
        if (gid < size) {
            a[gid] /= b[gid];
        }
    } else {
        if (gid < size) {
            int j = gid % broadcast_dim;
            a[gid] /= b[j];
        }
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void sum_along_dim(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> input,
    uniform RWStructuredBuffer<float> output,
    uniform int outerSize,
    uniform int reducedSize,
    uniform int innerSize
)
{
    int gid_outer = dispatchThreadID.x;
    int gid_inner = dispatchThreadID.y;

    if (gid_outer >= outerSize || gid_inner >= innerSize) return;

    float sum = 0.0f;
    for (int i = 0; i < reducedSize; i++) {
        int idx = gid_outer * reducedSize * innerSize + i * innerSize + gid_inner;
        sum += input[idx];
    }

    int resultIndex = gid_outer * innerSize + gid_inner;
    output[resultIndex] = sum;
}

[shader("compute")]
void softmax_last_dim(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> input,
    uniform RWStructuredBuffer<float> output,
    uniform int rows,
    uniform int lastDim,
    uniform float temperature
)
{
    int row = dispatchThreadID.x;
    if (row >= rows) return;
    int offset = row * lastDim;

    // look for max for numerical stability
    float max_val = input[offset];
    for (int i = 1; i < lastDim; i++) {
        float val = input[offset + i];
        if (val > max_val) max_val = val;
    }

    // compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < lastDim; i++) {
        float val = (input[offset + i] - max_val) / temperature;
        float exp_val = exp(val);
        output[offset + i] = exp_val;
        sum += exp_val;
    }

    // normalization
    for (int i = 0; i < lastDim; i++) {
        output[offset + i] /= sum;
    }
}
