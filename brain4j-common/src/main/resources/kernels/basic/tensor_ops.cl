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

__kernel void sub(
    __global float* a,
    __global const float* b,
    int size,
    int broadcast_dim,
    int batch
) {
    int gid = get_global_id(0);

    if (broadcast_dim == -1) {
        if (gid < size) {
            a[gid] -= b[gid];
        }
    } else {
        if (gid < size) {
            int j = gid % broadcast_dim;
            a[gid] -= b[j];
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

__kernel void div(
    __global float* a,
    __global const float* b,
    int size,
    int broadcast_dim,
    int batch
) {
    int gid = get_global_id(0);

    if (broadcast_dim == -1) {
        if (gid < size) {
            a[gid] /= b[gid];
        }
    } else {
        if (gid < size) {
            int j = gid % broadcast_dim;
            a[gid] /= b[j];
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

__kernel void sum_along_dim(
    __global const float* input,
    __global float* output,
    const int outerSize,
    const int reducedSize,
    const int innerSize
) {
    int gid_outer = get_global_id(0);
    int gid_inner = get_global_id(1);

    if (gid_outer >= outerSize || gid_inner >= innerSize) return;

    float sum = 0.0f;
    for (int i = 0; i < reducedSize; i++) {
        int idx = gid_outer * reducedSize * innerSize + i * innerSize + gid_inner;
        sum += input[idx];
    }

    int resultIndex = gid_outer * innerSize + gid_inner;
    output[resultIndex] = sum;
}

__kernel void layer_norm(
    __global float* data,
    const int batchSize,
    const int featuresSize,
    const float epsilon
) {
    int batch_idx = get_global_id(0);

    if (batch_idx >= batchSize) return;

    int offset = batch_idx * featuresSize;

    float mean = 0.0f;

    for (int j = 0; j < featuresSize; j++) {
        mean += data[offset + j];
    }

    mean /= featuresSize;

    float variance = 0.0f;

    for (int j = 0; j < featuresSize; j++) {
        float diff = data[offset + j] - mean;
        variance += diff * diff;
    }

    variance /= featuresSize;

    float denom = sqrt(variance + epsilon);

    for (int j = 0; j < featuresSize; j++) {
        data[offset + j] = (data[offset + j] - mean) / denom;
    }
}

__kernel void softmax_last_dim(
    __global const float* input,
    __global float* output,
    const int lastDim,
    const float temperature
) {
    int row = get_global_id(0);

    int offset = row * lastDim;

    float max_val = input[offset];
    for (int i = 1; i < lastDim; i++) {
        float val = input[offset + i];
        if (val > max_val) max_val = val;
    }

    float sum = 0.0f;
    for (int i = 0; i < lastDim; i++) {
        float val = (input[offset + i] - max_val) / temperature;
        float exp_val = exp(val);
        output[offset + i] = exp_val;
        sum += exp_val;
    }

    for (int i = 0; i < lastDim; i++) {
        output[offset + i] /= sum;
    }
}