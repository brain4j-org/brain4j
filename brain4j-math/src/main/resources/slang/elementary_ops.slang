// mask operation: data[i] = max(0, data[i] - mask[i])
[shader("compute")]
[numthreads(256, 1, 1)]
void mask(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> data,
    uniform StructuredBuffer<float> maskBuffer,
    uniform int length
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    data[gid] = max(0.0f, data[gid] - maskBuffer[gid]);
}

// add scalar: data[i] += value
[shader("compute")]
[numthreads(256, 1, 1)]
void add_scalar(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> data,
    uniform float value,
    uniform int length
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    data[gid] += value;
}

// sub scalar: data[i] -= value
[shader("compute")]
[numthreads(256, 1, 1)]
void sub_scalar(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> data,
    uniform float value,
    uniform int length
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    data[gid] -= value;
}

// mul scalar: data[i] *= value
[shader("compute")]
[numthreads(256, 1, 1)]
void mul_scalar(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> data,
    uniform float value,
    uniform int length
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    data[gid] *= value;
}

// div scalar: data[i] /= value
[shader("compute")]
[numthreads(256, 1, 1)]
void div_scalar(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> data,
    uniform float value,
    uniform int length
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    data[gid] /= value;
}

// pow scalar: data[i] = pow(data[i], value)
[shader("compute")]
[numthreads(256, 1, 1)]
void pow_scalar(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> data,
    uniform float value,
    uniform int length
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    data[gid] = pow(data[gid], value);
}

// sqrt: data[i] = sqrt(data[i])
[shader("compute")]
[numthreads(256, 1, 1)]
void sqrt_op(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> data,
    uniform int length
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    data[gid] = sqrt(data[gid]);
}

