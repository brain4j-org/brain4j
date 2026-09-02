// mask operation: data[i] = max(0, data[i] - mask[i])

// add scalar: data[i] += value

// sub scalar: data[i] -= value

// mul scalar: data[i] *= value

// div scalar: data[i] /= value

// pow scalar: data[i] = pow(data[i], value)

// sqrt: data[i] = sqrt(data[i])

[shader("compute")]
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
