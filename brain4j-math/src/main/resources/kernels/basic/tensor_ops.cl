#define TILE_SIZE 16
#define MAX_RANK 16

__kernel void layer_norm(
    __global const float* input,
    __global float* output,
    const int features,
    const int batchSize,
    const float epsilon
) {
    int batchIdx = get_global_id(0);
    if (batchIdx >= batchSize) return;

    int base = batchIdx * features;

    float mean = 0.0f;
    for (int j = 0; j < features; j++) {
        mean += input[base + j];
    }
    mean /= features;

    float var = 0.0f;
    for (int j = 0; j < features; j++) {
        float diff = input[base + j] - mean;
        var += diff * diff;
    }
    var /= features;

    float denom = sqrt(var + epsilon);

    for (int j = 0; j < features; j++) {
        output[base + j] = (input[base + j] - mean) / denom;
    }
}

__kernel void matmul_batched(
    __global const float* A,
    __global const float* B,
    __global float* C,
    __global const int* offsetsA,
    __global const int* offsetsB,
    __global const int* offsetsC,
    const int M,
    const int N,
    const int P,
    const int batchCount,
    const int transA,
    const int transB
) {
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    const int batch = get_global_id(2);

    const int local_row = get_local_id(0);
    const int local_col = get_local_id(1);

    if (batch >= batchCount) return;

    const int K = (transA == 0) ? N : M;

    const __global float* A_batch = A + offsetsA[batch];
    const __global float* B_batch = B + offsetsB[batch];
    __global float* C_batch = C + offsetsC[batch];

    const int A_rowStride = (transA == 0) ? N : M;
    const int A_colStride = 1;

    const int B_rowStride = (transB == 0) ? P : N;
    const int B_colStride = 1;

    __local float Asub[TILE_SIZE][TILE_SIZE];
    __local float Bsub[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        const int tiled_k = t * TILE_SIZE;

        const int a_r = row;
        const int a_c = tiled_k + local_col;
        const int b_r = tiled_k + local_row;
        const int b_c = col;

        if (a_r < M && a_c < K) {
            if (transA == 0) {
                Asub[local_row][local_col] = A_batch[a_r * A_rowStride + a_c * A_colStride];
            } else {
                Asub[local_row][local_col] = A_batch[a_c * A_rowStride + a_r * A_colStride];
            }
        } else {
            Asub[local_row][local_col] = 0.0f;
        }

        if (b_r < K && b_c < P) {
            if (transB == 0) {
                Bsub[local_row][local_col] = B_batch[b_r * B_rowStride + b_c * B_colStride];
            } else {
                Bsub[local_row][local_col] = B_batch[b_c * B_rowStride + b_r * B_colStride];
            }
        } else {
            Bsub[local_row][local_col] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += Asub[local_row][k] * Bsub[k][local_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < M && col < P) {
        C_batch[row * P + col] = sum;
    }
}

__kernel void matmul_legacy(
    __global const float* A,
    __global const float* B,
    __global float* C,
    __global const int* offsetsA,
    __global const int* offsetsB,
    __global const int* offsetsC,
    const int M,
    const int N,
    const int P,
    const int batchCount
) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    int batch = get_global_id(2);

    int local_row = get_local_id(0);
    int local_col = get_local_id(1);

    if (batch >= batchCount) return;

    bool valid = (row < M) && (col < P);

    const __global float* A_batch = A + offsetsA[batch];
    const __global float* B_batch = B + offsetsB[batch];
    __global float* C_batch = C + offsetsC[batch];

    __local float Asub[TILE_SIZE][TILE_SIZE];
    __local float Bsub[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        int tiled_col_A = t * TILE_SIZE + local_col;
        int tiled_row_B = t * TILE_SIZE + local_row;

        if (row < M && tiled_col_A < N) {
            Asub[local_row][local_col] = A_batch[row * N + tiled_col_A];
        } else {
            Asub[local_row][local_col] = 0.0f;
        }

        if (tiled_row_B < N && col < P) {
            Bsub[local_row][local_col] = B_batch[tiled_row_B * P + col];
        } else {
            Bsub[local_row][local_col] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += Asub[local_row][k] * Bsub[k][local_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (valid) {
        C_batch[row * P + col] = sum;
    }
}

__kernel void matmul(
    __global float* A,
    __global float* B,
    __global float* C,
    const int M,
    const int N,
    const int P,
    const int transA,
    const int transB
) {
    int tiled_row = get_global_id(0);
    int tiled_col = get_global_id(1);
    int local_row = get_local_id(0);
    int local_col = get_local_id(1);

    __local float Asub[TILE_SIZE][TILE_SIZE];
    __local float Bsub[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; ++t) {
        int tiled_col_A = t * TILE_SIZE + local_col;
        int tiled_row_B = t * TILE_SIZE + local_row;

        if (transA == 0) {
            if (tiled_row < M && tiled_col_A < N)
                Asub[local_row][local_col] = A[tiled_row * N + tiled_col_A];
            else
                Asub[local_row][local_col] = 0.0f;
        } else {
            if (tiled_col_A < M && tiled_row < N)
                Asub[local_row][local_col] = A[tiled_col_A * M + tiled_row];
            else
                Asub[local_row][local_col] = 0.0f;
        }

        if (transB == 0) {
            if (tiled_row_B < N && tiled_col < P)
                Bsub[local_row][local_col] = B[tiled_row_B * P + tiled_col];
            else
                Bsub[local_row][local_col] = 0.0f;
        } else {
            if (tiled_col < N && tiled_row_B < P)
                Bsub[local_row][local_col] = B[tiled_col * N + tiled_row_B];
            else
                Bsub[local_row][local_col] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k)
            sum += Asub[local_row][k] * Bsub[k][local_col];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tiled_row < M && tiled_col < P)
        C[tiled_row * P + tiled_col] = sum;

    /*__local float Asub[TILE_SIZE][TILE_SIZE];
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
        C[row * P + col] = sum;*/
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




__kernel void slice(
    __global const float* srcData,
    __global float* dstData,
    __global const int* srcStrides,
    __global const int* dstStrides,
    __global const int* dstShape,
    __global const int* starts,
    __global const int* steps,
    const int rank
) {
    int dstLinearIdx = get_global_id(0);

    int totalElements = 1;
    for (int i = 0; i < rank; i++) totalElements *= dstShape[i];
    if (dstLinearIdx >= totalElements) return;

    int tmp = dstLinearIdx;
    int srcOffset = 0;

    for (int i = 0; i < rank; i++) {
        int idx = tmp / dstStrides[i];
        tmp = tmp % dstStrides[i];

        int srcIdx = starts[i] + idx * steps[i];
        srcOffset += srcIdx * srcStrides[i];
    }

    dstData[dstLinearIdx] = srcData[srcOffset];
}

__kernel void broadcast(
    __global const float* in,
    __global float* out,
    __global const int* inShape,
    __global const int* outShape,
    __global const int* inStrides,
    const int rank,
    const int outSize
) {
    int dstLinearIdx = get_global_id(0);

    if (dstLinearIdx >= outSize) return;  // ← usa direttamente outSize

    int tmp = dstLinearIdx;
    int srcOffset = 0;

    for (int i = rank - 1; i >= 0; i--) {
        int idx = tmp % outShape[i];
        tmp = tmp / outShape[i];

        if (inShape[i] != 1) {srcOffset += idx * inStrides[i];   }
    }

    out[dstLinearIdx] = in[srcOffset];
}

__kernel void concat_last_dim(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int outerSize,
    const int lastA,
    const int lastB,
    const int concatLast
) {
    int gid = get_global_id(0);
    int total = outerSize * concatLast;
    if (gid >= total) return;

    int row = gid / concatLast;
    int col = gid % concatLast;

    if (col < lastA) {
        int aIdx = row * lastA + col;
        C[gid] = A[aIdx];
    } else {
        int bIdx = row * lastB + (col - lastA);
        C[gid] = B[bIdx];
    }
}

__kernel void concat_copy_a(
    __global const float* A,
    __global float* C,
    const int numBlocks,
    const int thisDim,
    const int otherDim,
    const int blockSize
) {
    int gid = get_global_id(0);
    int blockElem = thisDim * blockSize;
    int total = numBlocks * blockElem;
    if (gid >= total) return;

    int block = gid / blockElem;
    int inBlockIdx = gid % blockElem;

    int destStride = (thisDim + otherDim) * blockSize;
    int destIndex = block * destStride + inBlockIdx;

    C[destIndex] = A[gid];
}

__kernel void concat_copy_b(
    __global const float* B,
    __global float* C,
    const int numBlocks,
    const int thisDim,
    const int otherDim,
    const int blockSize
) {
    int gid = get_global_id(0);
    int blockElem = otherDim * blockSize;
    int total = numBlocks * blockElem;
    if (gid >= total) return;

    int block = gid / blockElem;
    int inBlockIdx = gid % blockElem;

    int destStride = (thisDim + otherDim) * blockSize;
    int destIndex = block * destStride + thisDim * blockSize + inBlockIdx;

    C[destIndex] = B[gid];
}
