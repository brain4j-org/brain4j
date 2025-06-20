__kernel void hard_clip(
    __global float* data,
    const float bound,
    const int length
) {
    int gid = get_global_id(0);
    if (gid < length) {
        data[gid] = fmax(-bound, fmin(bound, data[gid]));
    }
}

__kernel void l2_clip(
    __global float* data,
    const float scale,
    const int length
) {
    int gid = get_global_id(0);


}