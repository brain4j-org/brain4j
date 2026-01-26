[shader("compute")]
[numthreads(256, 1, 1)]
void elu_forward(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> input,
    uniform RWStructuredBuffer<float> output,
    uniform float alpha,
    uniform int length
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    float x = input[gid];
    output[gid] = x > 0.0f ? x : (alpha * (exp(x) - 1.0f));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void elu_backward(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> input,
    uniform RWStructuredBuffer<float> output,
    uniform float alpha,
    uniform int length
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    float x = input[gid];
    output[gid] = x > 0.0f ? 1.0f : alpha * exp(x);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void gelu_forward(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> input,
    uniform RWStructuredBuffer<float> output,
    uniform int length
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    float x = input[gid];
    float c = 0.044715f;
    float root2_over_pi = 0.7978845608f; // TODO: move to common, this is the value of sqrt(2/pi)
    float x3 = x * x * x;
    float tanh_val = tanh(root2_over_pi * (x + c * x3));
    output[gid] = 0.5f * x * (1.0f + tanh_val);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void gelu_backward(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> input,
    uniform RWStructuredBuffer<float> output,
    uniform int length
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    float x = input[gid];
    float c = 0.044715f;
    float root2_over_pi = 0.7978845608f;
    float x2 = x * x;
    float x3 = x2 * x;
    float tanh_arg = root2_over_pi * (x + c * x3);
    float tanh_val = tanh(tanh_arg);
    float sech2 = 1.0f - tanh_val * tanh_val;
    float term1 = 0.5f * (1.0f + tanh_val);
    float term2 = 0.5f * x * sech2 * root2_over_pi * (1.0f + 3.0f * c * x2);
    output[gid] = term1 + term2;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void leakyrelu_forward(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> input,
    uniform RWStructuredBuffer<float> output,
    uniform float alpha,
    uniform int length
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    float x = input[gid];
    output[gid] = x > 0.0f ? x : alpha * x;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void leakyrelu_backward(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> input,
    uniform RWStructuredBuffer<float> output,
    uniform float alpha,
    uniform int length
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    float x = input[gid];
    output[gid] = x > 0.0f ? 1.0f : alpha;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void linear_forward(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> input,
    uniform RWStructuredBuffer<float> output,
    uniform int length
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    output[gid] = input[gid];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void linear_backward(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> input,
    uniform RWStructuredBuffer<float> output,
    uniform int length
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    output[gid] = 1.0f;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mish_forward(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> input,
    uniform RWStructuredBuffer<float> output,
    uniform int length
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    float x = input[gid];
    float sp = log(1.0f + exp(x));
    output[gid] = x * tanh(sp);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mish_backward(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> input,
    uniform RWStructuredBuffer<float> output,
    uniform int length
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    float x = input[gid];
    float e = exp(x);
    float om = log(1.0f + e);
    float tanh_om = tanh(om);
    float sech2 = 1.0f - tanh_om * tanh_om;
    output[gid] = tanh_om + x * e * sech2 / (1.0f + e);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void relu_forward(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> input,
    uniform RWStructuredBuffer<float> output,
    uniform int length
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    float x = input[gid];
    output[gid] = max(x, 0.0f);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void relu_backward(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> input,
    uniform RWStructuredBuffer<float> output,
    uniform int length
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    float x = input[gid];
    output[gid] = x > 0.0f ? 1.0f : 0.0f;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sigmoid_forward(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> input,
    uniform RWStructuredBuffer<float> output,
    uniform int length
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    float x = input[gid];
    output[gid] = 1.0f / (1.0f + exp(-x));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sigmoid_backward(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> input,
    uniform RWStructuredBuffer<float> output,
    uniform int length
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    float x = input[gid];
    float activated = 1.0f / (1.0f + exp(-x));
    output[gid] = activated * (1.0f - activated);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void softmax_forward(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> input,
    uniform RWStructuredBuffer<float> output,
    uniform int length
)
{
    // TODO: implement row-wise softmax
}

[shader("compute")]
[numthreads(256, 1, 1)]
void softmax_backward(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> input,
    uniform RWStructuredBuffer<float> output,
    uniform int length
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    float x = input[gid];
    output[gid] = x * (1.0f - x);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void swish_forward(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> input,
    uniform RWStructuredBuffer<float> output,
    uniform int length
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    float x = input[gid];
    float sigmoid_x = 1.0f / (1.0f + exp(-x));
    output[gid] = x * sigmoid_x;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void swish_backward(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> input,
    uniform RWStructuredBuffer<float> output,
    uniform int length
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    float x = input[gid];
    float sigmoid_x = 1.0f / (1.0f + exp(-x));
    output[gid] = sigmoid_x + x * sigmoid_x * (1.0f - sigmoid_x);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void softplus_forward(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> input,
    uniform RWStructuredBuffer<float> output,
    uniform int length
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    float x = input[gid];
    // log1p prevents numerical instability when exp(x) << arbitrary value
    output[gid] = log(1.0f + exp(x));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void softplus_backward(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> input,
    uniform RWStructuredBuffer<float> output,
    uniform int length
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    float x = input[gid];
    // TODO: check for numerical stability
    if (x >= 0.0f) {
        float expNeg = exp(-x);
        output[gid] = 1.0f / (1.0f + expNeg);
    } else {
        float expPos = exp(x);
        output[gid] = expPos / (1.0f + expPos);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void tanh_forward(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> input,
    uniform RWStructuredBuffer<float> output,
    uniform int length
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    float x = input[gid];
    output[gid] = tanh(x);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void tanh_backward(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> input,
    uniform RWStructuredBuffer<float> output,
    uniform int length
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    float x = input[gid];
    float t = tanh(x);
    output[gid] = 1.0f - t * t;
}

