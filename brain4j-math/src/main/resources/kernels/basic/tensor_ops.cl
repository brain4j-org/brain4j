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

__kernel void matmul_unoptimized(
    __global float* A,
    __global float* B,
    __global float* C,
    const int m,
    const int n,
    const int p
) {
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    
    const int local_row = get_local_id(0);
    const int local_col = get_local_id(1);
    
    const int block_dim_x = get_local_size(0);
    const int block_dim_y = get_local_size(1);
    
    if (row < m && col < p) {
        float sum = 0.0f;
        
        int k = 0;
        
        for (; k + 3 < n; k += 4) {
            sum += A[row * n + k] * B[k * p + col];
            sum += A[row * n + k + 1] * B[(k + 1) * p + col];
            sum += A[row * n + k + 2] * B[(k + 2) * p + col];
            sum += A[row * n + k + 3] * B[(k + 3) * p + col];
        }
        
        for (; k < n; k++) {
            sum += A[row * n + k] * B[k * p + col];
        }
        
        C[row * p + col] = sum;
    }
}

__kernel void element_wise_add(
    __global float* A,
    __global float* B,
    __global float* C,
    const int size
) {
    const int i = get_global_id(0);
    
    if (i + 3 < size && (i % 4) == 0) {
        float4 a_vec = vload4(i/4, A);
        float4 b_vec = vload4(i/4, B);
        
        float4 c_vec = a_vec + b_vec;
        
        vstore4(c_vec, i/4, C);
    }
    else if (i < size) {
        C[i] = A[i] + B[i];
    }
}

__kernel void element_wise_mul(
    __global float* A,
    __global float* B,
    __global float* C,
    const int size
) {
    const int i = get_global_id(0);
    
    if (i + 3 < size && (i % 4) == 0) {
        float4 a_vec = vload4(i/4, A);
        float4 b_vec = vload4(i/4, B);
        
        float4 c_vec = a_vec * b_vec;
        
        vstore4(c_vec, i/4, C);
    }
    else if (i < size) {
        C[i] = A[i] * B[i];
    }
}