__kernel void elu_forward(
    __global const float* input,
    __global float* output,
    const float alpha,
    const int length
) {
    int gid = get_global_id(0);
    if (gid < length) {
        float x = input[gid];
        output[gid] = x > 0.0f ? x : (alpha * (exp(x) - 1.0f));
    }
}

__kernel void elu_backward(
    __global const float* input,
    __global float* output,
    const float alpha,
    const int length
) {
    int gid = get_global_id(0);
    if (gid < length) {
       float x = input[gid];
       output[gid] = x > 0.0f ? 1.0f : alpha * exp(x);
    }
}

__kernel void gelu_forward(
    __global const float* input,
    __global float* output,
    const int length
) {
    int gid = get_global_id(0);
    if (gid < length) {
        float x = input[gid];
        float c = 0.044715f;
        float root2_over_pi = 0.7978845608f;
        float x3 = x * x * x;
        float tanh_val = tanh(root2_over_pi * (x + c * x3));
        output[gid] = 0.5f * x * (1.0f + tanh_val);
    }
}

__kernel void gelu_backward(
    __global const float* input,
    __global float* output,
    const int length
) {
    int gid = get_global_id(0);
    if (gid < length) {
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
}

__kernel void leakyrelu_forward(
    __global const float* input,
    __global float* output,
    const float alpha,
    const int length
) {
    int gid = get_global_id(0);
    if (gid < length) {
        float x = input[gid];
        output[gid] = x > 0.0f ? x : alpha * x;
    }
}

__kernel void leakyrelu_backward(
    __global const float* input,
    __global float* output,
    const float alpha,
    const int length
) {
    int gid = get_global_id(0);
    if (gid < length) {
        float x = input[gid];
        output[gid] = x > 0.0f ? 1.0f : alpha;
    }
}

__kernel void linear_forward(
    __global const float* input,
    __global float* output,
    const int length
) {
    int gid = get_global_id(0);
    if (gid < length) {
        output[gid] = input[gid];
    }
}

__kernel void linear_backward(
    __global const float* input,
    __global float* output,
    const int length
) {
    int gid = get_global_id(0);
    if (gid < length) {
        output[gid] = 1.0f;
    }
}

__kernel void mish_forward(
    __global const float* input,
    __global float* output,
    const int length
) {
    int gid = get_global_id(0);
    if (gid < length) {
        float x = input[gid];
        float sp = log(1.0f + exp(x));
        output[gid] = x * tanh(sp);
    }
}

__kernel void mish_backward(
    __global const float* input,
    __global float* output,
    const int length
) {
    int gid = get_global_id(0);
    if (gid < length) {
        float x = input[gid];
        float e = exp(x);
        float om = log(1.0f + e);
        float tanh_om = tanh(om);
        float sech2 = 1.0f - tanh_om * tanh_om;
        output[gid] = tanh_om + x * e * sech2 / (1.0f + e);
    }
}

__kernel void relu_forward(
    __global const float* input,
    __global float* output,
    const int length
) {
    int gid = get_global_id(0);
    if (gid < length) {
        float x = input[gid];
        output[gid] = fmax(x, 0.0f);
    }
}

__kernel void relu_backward(
    __global const float* input,
    __global float* output,
    const int length
) {
    int gid = get_global_id(0);
    if (gid < length) {
        float x = input[gid];
        output[gid] = x > 0.0f ? 1.0f : 0.0f;
    }
}

__kernel void sigmoid_forward(
    __global const float* input,
    __global float* output,
    const int length
) {
    int gid = get_global_id(0);
    if (gid < length) {
        float x = input[gid];
        output[gid] = 1.0f / (1.0f + exp(-x));
    }
}

__kernel void sigmoid_backward(
    __global const float* input,
    __global float* output,
    const int length
) {
    int gid = get_global_id(0);
    if (gid < length) {
        float x = input[gid];
        float activated = 1.0f / (1.0f + exp(-x));
        output[gid] = activated * (1 - activated);
    }
}

__kernel void softmax_forward(
    __global const float* input,
    __global float* output,
    const int length
) {
}

__kernel void softmax_backward(
    __global const float* input,
    __global float* output,
    const int length
) {
    int gid = get_global_id(0);
    if (gid < length) {
        float x = input[gid];
        output[gid] = x * (1.0 - x);
    }
}

__kernel void swish_forward(
    __global const float* input,
    __global float* output,
    const int length
) {
    int gid = get_global_id(0);
    if (gid < length) {
        float x = input[gid];
        float sigmoid = 1.0f / (1.0f + exp(-x));
        output[gid] = x * sigmoid;
    }
}

__kernel void swish_backward(
    __global const float* input,
    __global float* output,
    const int length
) {
    int gid = get_global_id(0);
    if (gid < length) {
        float x = input[gid];
        float sigmoid = 1.0f / (1.0f + exp(-x));
        output[gid] = sigmoid + x * sigmoid * (1.0f - sigmoid);
    }
}

__kernel void tanh_forward(
    __global const float* input,
    __global float* output,
    const int length
) {
    int gid = get_global_id(0);
    if (gid < length) {
        float x = input[gid];
        output[gid] = tanh(x);
    }
}

__kernel void tanh_backward(
    __global const float* input,
    __global float* output,
    const int length
) {
    int gid = get_global_id(0);
    if (gid < length) {
        float x = input[gid];
        float t = tanh(x);
        output[gid] = 1.0f - t * t;
    }
}