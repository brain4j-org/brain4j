#define TILE_SIZE 16

__kernel void matmul(
    __global float* A,
    __global float* B,
    __global float* C,
    const int M,
    const int N,
    const int P
) {
    __local float Asub[TILE_SIZE][TILE_SIZE];
    __local float Bsub[TILE_SIZE][TILE_SIZE];

    int row = get_global_id(0);
    int col = get_global_id(1);
    int local_row = get_local_id(0);
    int local_col = get_local_id(1);

    float sum = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int tiled_row = row;
        int tiled_col = t * TILE_SIZE + local_col;
        if (tiled_row < M && tiled_col < N)
            Asub[local_row][local_col] = A[tiled_row * N + tiled_col];
        else
            Asub[local_row][local_col] = 0.0f;

        tiled_row = t * TILE_SIZE + local_row;
        tiled_col = col;
        if (tiled_row < N && tiled_col < P)
            Bsub[local_row][local_col] = B[tiled_row * P + tiled_col];
        else
            Bsub[local_row][local_col] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += Asub[local_row][k] * Bsub[k][local_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < M && col < P)
        C[row * P + col] = sum;
}

__kernel void add(
    __global float* a,
    __global const float* b,
    int size,
    int broadcast_dim,
    int batch
) {
    int gid = get_global_id(0);

    if (broadcast_dim == -1) {
        if (gid < size) {
            a[gid] += b[gid];
        }
    } else {
        if (gid < size) {
            int j = gid % broadcast_dim;
            a[gid] += b[j];
        }
    }
}

__kernel void mul(
    __global float* a,
    __global const float* b,
    int size,
    int broadcast_dim,
    int batch
) {
    int gid = get_global_id(0);

    if (broadcast_dim == -1) {
        if (gid < size) {
            a[gid] *= b[gid];
        }
    } else {
        if (gid < size) {
            int j = gid % broadcast_dim;
            a[gid] *= b[j];
        }
    }
}

__kernel void transpose(
    __global const float* input,
    __global float* output,
    const int rows,
    const int cols,
    const int inRowStride,
    const int inColStride,
    const int outRowStride,
    const int outColStride
) {
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < rows && col < cols) {
        int inputIndex = row * inRowStride + col * inColStride;
        int outputIndex = col * outRowStride + row * outColStride;
        output[outputIndex] = input[inputIndex];
    }
}