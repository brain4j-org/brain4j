// fill: data[i] = value
[shader("compute")]
void fill_const(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> data,
    uniform float value,
    uniform int length
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    data[gid] = value;
}
