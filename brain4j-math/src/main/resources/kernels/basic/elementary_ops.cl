__kernel void add_scalar(__global float* data, const float value, const int length) {
    int i = get_global_id(0);
    if (i < length) {
        data[i] += value;
    }
}

__kernel void sub_scalar(__global float* data, const float value, const int length) {
    int i = get_global_id(0);
    if (i < length) {
        data[i] -= value;
    }
}

__kernel void mul_scalar(__global float* data, const float value, const int length) {
    int i = get_global_id(0);
    if (i < length) {
        data[i] *= value;
    }
}

__kernel void div_scalar(__global float* data, const float value, const int length) {
    int i = get_global_id(0);
    if (i < length) {
        data[i] /= value;
    }
}

__kernel void pow_scalar(__global float* data, const float value, const int length) {
    int i = get_global_id(0);
    if (i < length) {
        data[i] = pow(data[i], value);
    }
}

__kernel void sqrt(__global float* data, const int length) {
    int i = get_global_id(0);
    if (i < length) {
        data[i] = sqrt(data[i]);
    }
}