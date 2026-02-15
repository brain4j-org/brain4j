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

[shader("compute")]
[numthreads(256,1,1)]
void activation_forward(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform int activationType,
    uniform float alpha,
    uniform int length,
    StructuredBuffer<float> input,
    RWStructuredBuffer<float> output
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    float x = input[gid];
    float result = x;

    switch (activationType)
    {
        case ACT_RELU:
            result = max(x, 0.0f);
            break;

        case ACT_LEAKYRELU:
            result = x > 0.0f ? x : alpha * x;
            break;

        case ACT_ELU:
            result = x > 0.0f ? x : alpha * (exp(x) - 1.0f);
            break;

        case ACT_SIGMOID:
            result = 1.0f / (1.0f + exp(-x));
            break;

        case ACT_TANH:
            result = tanh(x);
            break;

        case ACT_SWISH:
        {
            float s = 1.0f / (1.0f + exp(-x));
            result = x * s;
            break;
        }

        case ACT_SOFTPLUS:
            result = log(1.0f + exp(x));
            break;

        case ACT_GELU:
        {
            float c = 0.044715f;
            float root2_over_pi = 0.7978845608f;
            float x3 = x * x * x;
            float t = tanh(root2_over_pi * (x + c * x3));
            result = 0.5f * x * (1.0f + t);
            break;
        }

        case ACT_MISH:
        {
            float sp = log(1.0f + exp(x));
            result = x * tanh(sp);
            break;
        }

        case ACT_LINEAR:
            result = x;
            break;
    }

    output[gid] = result;
}


[shader("compute")]
[numthreads(256,1,1)]
void activation_backward(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform int activationType,
    uniform float alpha,
    uniform int length,
    StructuredBuffer<float> input,
    RWStructuredBuffer<float> output
)
{
    uint gid = dispatchThreadID.x;
    if (gid >= length) return;

    float x = input[gid];
    float result = 1.0f;

    switch (activationType)
    {
        case ACT_RELU:
            result = x > 0.0f ? 1.0f : 0.0f;
            break;

        case ACT_LEAKYRELU:
            result = x > 0.0f ? 1.0f : alpha;
            break;

        case ACT_ELU:
            result = x > 0.0f ? 1.0f : alpha * exp(x);
            break;

        case ACT_SIGMOID:
        {
            float s = 1.0f / (1.0f + exp(-x));
            result = s * (1.0f - s);
            break;
        }

        case ACT_TANH:
        {
            float t = tanh(x);
            result = 1.0f - t * t;
            break;
        }

        case ACT_SWISH:
        {
            float s = 1.0f / (1.0f + exp(-x));
            result = s + x * s * (1.0f - s);
            break;
        }

        case ACT_SOFTPLUS:
        {
            if (x >= 0.0f)
            {
                float expNeg = exp(-x);
                result = 1.0f / (1.0f + expNeg);
            }
            else
            {
                float expPos = exp(x);
                result = expPos / (1.0f + expPos);
            }
            break;
        }

        case ACT_GELU:
        {
            float c = 0.044715f;
            float r = 0.7978845608f;
            float x2 = x * x;
            float x3 = x2 * x;

            float tanh_arg = r * (x + c * x3);
            float t = tanh(tanh_arg);
            float sech2 = 1.0f - t * t;

            float term1 = 0.5f * (1.0f + t);
            float term2 = 0.5f * x * sech2 * r * (1.0f + 3.0f * c * x2);

            result = term1 + term2;
            break;
        }

        case ACT_MISH:
        {
            float e = exp(x);
            float om = log(1.0f + e);
            float t = tanh(om);
            float sech2 = 1.0f - t * t;

            result = t + x * e * sech2 / (1.0f + e);
            break;
        }

        case ACT_LINEAR:
            result = 1.0f;
            break;
    }

    output[gid] = result;
}
