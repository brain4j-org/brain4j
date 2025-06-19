__kernel void sigmoid_forward(__global const float* input, __global float* output, const int length) {
    int gid = get_global_id(0);
    if (gid < length) {
        float x = input[gid];
        output[gid] = 1.0f / (1.0f + exp(-x));
    }
}

__kernel void sigmoid_derivate(__global const float* activated, __global float* output, const int length) {
    int gid = get_global_id(0);
    if (gid < length) {
        float x = activated[gid];
        output[gid] = x * (1 - x);
    }
}

__kernel void relu_forward(__global const float* input, __global float* output, const int length) {
    int gid = get_global_id(0);
    if (gid < length) {
        float x = input[gid];
        output[gid] = fmax(x, 0.0f);
    }
}

__kernel void relu_derivative(__global const float* activated, __global float* output, const int length) {
    int gid = get_global_id(0);
    if (gid < length) {
        float x = activated[gid];
        output[gid] = x > 0.0f ? 1.0f : 0.0f;
    }
}