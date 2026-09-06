static const int ACT_RELU      = 0;
static const int ACT_LEAKYRELU = 1;
static const int ACT_ELU       = 2;
static const int ACT_SIGMOID   = 3;
static const int ACT_TANH      = 4;
static const int ACT_SWISH     = 5;
static const int ACT_SOFTPLUS  = 6;
static const int ACT_GELU      = 7;
static const int ACT_MISH      = 8;
static const int ACT_LINEAR    = 9;

float gelu_forward_value(float x)
{
    float c = 0.044715f;
    float root2_over_pi = 0.7978845608f;
    float x3 = x * x * x;
    float tanh_val = tanh(root2_over_pi * (x + c * x3));
    return 0.5f * x * (1.0f + tanh_val);
}

float gelu_backward_value(float x)
{
    float c = 0.044715f;
    float root2_over_pi = 0.7978845608f;
    float x2 = x * x;
    float x3 = x2 * x;
    float tanh_arg = root2_over_pi * (x + c * x3);
    float tanh_val = tanh(tanh_arg);
    float sech2 = 1.0f - tanh_val * tanh_val;
    float term1 = 0.5f * (1.0f + tanh_val);
    float term2 = 0.5f * x * sech2 * root2_over_pi * (1.0f + 3.0f * c * x2);
    return term1 + term2;
}

float sigmoid_value(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

float softplus_value(float x)
{
    if (x > 20.0f) return x;
    if (x < -20.0f) return exp(x);
    return log(1.0f + exp(x));
}

float activation_forward_value(int activationType, float alpha, float x)
{
    switch (activationType)
    {
        case ACT_RELU:
            return max(x, 0.0f);
        case ACT_LEAKYRELU:
            return x > 0.0f ? x : alpha * x;
        case ACT_ELU:
            return x > 0.0f ? x : alpha * (exp(x) - 1.0f);
        case ACT_SIGMOID:
            return sigmoid_value(x);
        case ACT_TANH:
            return tanh(x);
        case ACT_SWISH:
        {
            float s = sigmoid_value(x);
            return x * s;
        }
        case ACT_SOFTPLUS:
            return softplus_value(x);
        case ACT_GELU:
            return gelu_forward_value(x);
        case ACT_MISH:
            return x * tanh(softplus_value(x));
        case ACT_LINEAR:
            return x;
    }

    return x;
}

float activation_backward_value(int activationType, float alpha, float x)
{
    switch (activationType)
    {
        case ACT_RELU:
            return x > 0.0f ? 1.0f : 0.0f;
        case ACT_LEAKYRELU:
            return x > 0.0f ? 1.0f : alpha;
        case ACT_ELU:
            return x > 0.0f ? 1.0f : alpha * exp(x);
        case ACT_SIGMOID:
        {
            float s = sigmoid_value(x);
            return s * (1.0f - s);
        }
        case ACT_TANH:
        {
            float t = tanh(x);
            return 1.0f - t * t;
        }
        case ACT_SWISH:
        {
            float s = sigmoid_value(x);
            return s + x * s * (1.0f - s);
        }
        case ACT_SOFTPLUS:
            return sigmoid_value(x);
        case ACT_GELU:
            return gelu_backward_value(x);
        case ACT_MISH:
        {
            float sp = softplus_value(x);
            float t = tanh(sp);
            float sech2 = 1.0f - t * t;
            return t + x * sigmoid_value(x) * sech2;
        }
        case ACT_LINEAR:
            return 1.0f;
    }

    return 1.0f;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void activation_forward(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform int activationType,
    uniform float alpha,
    uniform int length,
    uniform StructuredBuffer<float> input,
    uniform RWStructuredBuffer<float> output
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    output[gid] = activation_forward_value(activationType, alpha, input[gid]);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void activation_backward(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform int activationType,
    uniform float alpha,
    uniform int length,
    uniform StructuredBuffer<float> input,
    uniform RWStructuredBuffer<float> output
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    output[gid] = activation_backward_value(activationType, alpha, input[gid]);
}

void activation_forward_named(
    uint3 dispatchThreadID,
    int activationType,
    float alpha,
    StructuredBuffer<float> input,
    RWStructuredBuffer<float> output,
    int length
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    output[gid] = activation_forward_value(activationType, alpha, input[gid]);
}

void activation_backward_named(
    uint3 dispatchThreadID,
    int activationType,
    float alpha,
    StructuredBuffer<float> input,
    RWStructuredBuffer<float> output,
    int length
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    output[gid] = activation_backward_value(activationType, alpha, input[gid]);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void elu_forward(uint3 id : SV_DispatchThreadID, uniform StructuredBuffer<float> input, uniform RWStructuredBuffer<float> output, uniform float alpha, uniform int length)
{
    activation_forward_named(id, ACT_ELU, alpha, input, output, length);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void elu_backward(uint3 id : SV_DispatchThreadID, uniform StructuredBuffer<float> input, uniform RWStructuredBuffer<float> output, uniform float alpha, uniform int length)
{
    activation_backward_named(id, ACT_ELU, alpha, input, output, length);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void gelu_forward(uint3 id : SV_DispatchThreadID, uniform StructuredBuffer<float> input, uniform RWStructuredBuffer<float> output, uniform int length)
{
    activation_forward_named(id, ACT_GELU, 0.0f, input, output, length);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void gelu_backward(uint3 id : SV_DispatchThreadID, uniform StructuredBuffer<float> input, uniform RWStructuredBuffer<float> output, uniform int length)
{
    activation_backward_named(id, ACT_GELU, 0.0f, input, output, length);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void leakyrelu_forward(uint3 id : SV_DispatchThreadID, uniform StructuredBuffer<float> input, uniform RWStructuredBuffer<float> output, uniform float alpha, uniform int length)
{
    activation_forward_named(id, ACT_LEAKYRELU, alpha, input, output, length);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void leakyrelu_backward(uint3 id : SV_DispatchThreadID, uniform StructuredBuffer<float> input, uniform RWStructuredBuffer<float> output, uniform float alpha, uniform int length)
{
    activation_backward_named(id, ACT_LEAKYRELU, alpha, input, output, length);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void linear_forward(uint3 id : SV_DispatchThreadID, uniform StructuredBuffer<float> input, uniform RWStructuredBuffer<float> output, uniform int length)
{
    activation_forward_named(id, ACT_LINEAR, 0.0f, input, output, length);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void linear_backward(uint3 id : SV_DispatchThreadID, uniform StructuredBuffer<float> input, uniform RWStructuredBuffer<float> output, uniform int length)
{
    activation_backward_named(id, ACT_LINEAR, 0.0f, input, output, length);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mish_forward(uint3 id : SV_DispatchThreadID, uniform StructuredBuffer<float> input, uniform RWStructuredBuffer<float> output, uniform int length)
{
    activation_forward_named(id, ACT_MISH, 0.0f, input, output, length);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mish_backward(uint3 id : SV_DispatchThreadID, uniform StructuredBuffer<float> input, uniform RWStructuredBuffer<float> output, uniform int length)
{
    activation_backward_named(id, ACT_MISH, 0.0f, input, output, length);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void relu_forward(uint3 id : SV_DispatchThreadID, uniform StructuredBuffer<float> input, uniform RWStructuredBuffer<float> output, uniform int length)
{
    activation_forward_named(id, ACT_RELU, 0.0f, input, output, length);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void relu_backward(uint3 id : SV_DispatchThreadID, uniform StructuredBuffer<float> input, uniform RWStructuredBuffer<float> output, uniform int length)
{
    activation_backward_named(id, ACT_RELU, 0.0f, input, output, length);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sigmoid_forward(uint3 id : SV_DispatchThreadID, uniform StructuredBuffer<float> input, uniform RWStructuredBuffer<float> output, uniform int length)
{
    activation_forward_named(id, ACT_SIGMOID, 0.0f, input, output, length);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sigmoid_backward(uint3 id : SV_DispatchThreadID, uniform StructuredBuffer<float> input, uniform RWStructuredBuffer<float> output, uniform int length)
{
    activation_backward_named(id, ACT_SIGMOID, 0.0f, input, output, length);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void softmax_forward(uint3 id : SV_DispatchThreadID, uniform StructuredBuffer<float> input, uniform RWStructuredBuffer<float> output, uniform int length)
{
    float maxVal = input[0];
    for (int i = 1; i < length; i++) {
        maxVal = max(maxVal, input[i]);
    }

    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
        float v = exp(input[i] - maxVal);
        output[i] = v;
        sum += v;
    }

    for (int i = 0; i < length; i++) {
        output[i] /= sum;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void softmax_backward(uint3 id : SV_DispatchThreadID, uniform StructuredBuffer<float> input, uniform RWStructuredBuffer<float> output, uniform int length)
{
    uint gid = id.x;
    if (gid >= length) return;

    float y = input[gid];
    output[gid] = y * (1.0f - y);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void softplus_forward(uint3 id : SV_DispatchThreadID, uniform StructuredBuffer<float> input, uniform RWStructuredBuffer<float> output, uniform int length)
{
    activation_forward_named(id, ACT_SOFTPLUS, 0.0f, input, output, length);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void softplus_backward(uint3 id : SV_DispatchThreadID, uniform StructuredBuffer<float> input, uniform RWStructuredBuffer<float> output, uniform int length)
{
    activation_backward_named(id, ACT_SOFTPLUS, 0.0f, input, output, length);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void swish_forward(uint3 id : SV_DispatchThreadID, uniform StructuredBuffer<float> input, uniform RWStructuredBuffer<float> output, uniform int length)
{
    activation_forward_named(id, ACT_SWISH, 0.0f, input, output, length);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void swish_backward(uint3 id : SV_DispatchThreadID, uniform StructuredBuffer<float> input, uniform RWStructuredBuffer<float> output, uniform int length)
{
    activation_backward_named(id, ACT_SWISH, 0.0f, input, output, length);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void tanh_forward(uint3 id : SV_DispatchThreadID, uniform StructuredBuffer<float> input, uniform RWStructuredBuffer<float> output, uniform int length)
{
    activation_forward_named(id, ACT_TANH, 0.0f, input, output, length);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void tanh_backward(uint3 id : SV_DispatchThreadID, uniform StructuredBuffer<float> input, uniform RWStructuredBuffer<float> output, uniform int length)
{
    activation_backward_named(id, ACT_TANH, 0.0f, input, output, length);
}
