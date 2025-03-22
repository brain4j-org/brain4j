__kernel void matmul(
    __global const float* A,
    __global const float* B,
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
    __global const float* A,
    __global const float* B,
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
    __global const float* A,
    __global const float* B,
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