static const int FA_TILE_SIZE = 16;
static const int FA_HEAD_DIM = 64;

// TODO: REWRITE THIS
[shader("compute")]
[numthreads(256, 1, 1)]
void flash_attention_forward(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> Q,
    uniform StructuredBuffer<float> K,
    uniform StructuredBuffer<float> V,
    uniform RWStructuredBuffer<float> O,
    uniform StructuredBuffer<int> qStrides,
    uniform StructuredBuffer<int> kStrides,
    uniform StructuredBuffer<int> vStrides,
    uniform StructuredBuffer<int> oStrides,
    uniform int B,
    uniform int H,
    uniform int L,
    uniform int D,
    uniform float scale,
    uniform int causal
)
{
    // global work layout: [L, B*H]
    int i = dispatchThreadID.x;   // query index in sequence [0..L)
    int bh = dispatchThreadID.y;  // combined batch*head index [0..B*H)

    if (i >= L || bh >= B * H) return;

    // we decode the batch and head indices
    int b = bh / H;
    int h = bh - b * H;

    // let's calculate now the base offsets using strides
    int qBase = b * qStrides[0] + h * qStrides[1] + i * qStrides[2];
    int oBase = b * oStrides[0] + h * oStrides[1] + i * oStrides[2];

    // then we initialize the output accumulator to 0
    for (int d = 0; d < D; ++d) {
        O[oBase + d * oStrides[3]] = 0.0f;
    }

    // online softmax state
    float m = -1e30f/*-INFINITY*/;  // running max
    float s = 0.0f;    // running sum

    // K-V pairs iteration
    for (int j = 0; j < L; ++j) {
        if (causal != 0 && j > i) continue;  // causal mask

        int kBase = b * kStrides[0] + h * kStrides[1] + j * kStrides[2];
        int vBase = b * vStrides[0] + h * vStrides[1] + j * vStrides[2];

        // compute dot(Q[i], K[j])
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) {
            dot += Q[qBase + d * qStrides[3]] * K[kBase + d * kStrides[3]];
        }

        float score = dot * scale;
        float new_m = max(m, score);
        float e1 = (m < -1e29f) ? 0.0f : exp(m - new_m);
        float e2 = exp(score - new_m);

        // then we update numerator accumulator: O = O * e1 + V[j] * e2
        for (int d = 0; d < D; ++d) {
            float old = O[oBase + d * oStrides[3]];
            float v = V[vBase + d * vStrides[3]];
            O[oBase + d * oStrides[3]] = old * e1 + v * e2;
        }

        s = s * e1 + e2;
        m = new_m;
    }

    // once we have the output we normalize it
    float inv_s = (s > 0.0f) ? (1.0f / s) : 0.0f;
    for (int d = 0; d < D; ++d) {
        O[oBase + d * oStrides[3]] *= inv_s;
    }
}

// ============================================================================
// FlashAttention Forward Pass with LSE Output
// ============================================================================

[shader("compute")]
[numthreads(256, 1, 1)]
void flash_attention_forward_with_lse(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> Q,
    uniform StructuredBuffer<float> K,
    uniform StructuredBuffer<float> V,
    uniform RWStructuredBuffer<float> O,
    uniform RWStructuredBuffer<float> LSE,
    uniform StructuredBuffer<int> qStrides,
    uniform StructuredBuffer<int> kStrides,
    uniform StructuredBuffer<int> vStrides,
    uniform StructuredBuffer<int> oStrides,
    uniform int B,
    uniform int H,
    uniform int L,
    uniform int D,
    uniform float scale,
    uniform int causal
)
{
    int i = dispatchThreadID.x;
    int bh = dispatchThreadID.y;

    if (i >= L || bh >= B * H) return;

    int b = bh / H;
    int h = bh - b * H;

    int qBase = b * qStrides[0] + h * qStrides[1] + i * qStrides[2];
    int oBase = b * oStrides[0] + h * oStrides[1] + i * oStrides[2];
    int lseIdx = b * H * L + h * L + i;

    for (int d = 0; d < D; ++d) {
        O[oBase + d * oStrides[3]] = 0.0f;
    }

    float m = -1e30f;
    float s = 0.0f;

    for (int j = 0; j < L; ++j) {
        if (causal != 0 && j > i) continue;

        int kBase = b * kStrides[0] + h * kStrides[1] + j * kStrides[2];
        int vBase = b * vStrides[0] + h * vStrides[1] + j * vStrides[2];

        float dot = 0.0f;
        for (int d = 0; d < D; ++d) {
            dot += Q[qBase + d * qStrides[3]] * K[kBase + d * kStrides[3]];
        }

        float score = dot * scale;
        float new_m = max(m, score);
        float e1 = (m < -1e29f) ? 0.0f : exp(m - new_m);
        float e2 = exp(score - new_m);

        for (int d = 0; d < D; ++d) {
            float old = O[oBase + d * oStrides[3]];
            float v = V[vBase + d * vStrides[3]];
            O[oBase + d * oStrides[3]] = old * e1 + v * e2;
        }

        s = s * e1 + e2;
        m = new_m;
    }

    // LSE = m + log(s) for backward pass
    LSE[lseIdx] = m + log(s + 1e-10f);

    float inv_s = (s > 0.0f) ? (1.0f / s) : 0.0f;
    for (int d = 0; d < D; ++d) {
        O[oBase + d * oStrides[3]] *= inv_s;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void flash_attention_backward(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> Q,
    uniform StructuredBuffer<float> K,
    uniform StructuredBuffer<float> V,
    uniform StructuredBuffer<float> O,
    uniform StructuredBuffer<float> dO,
    uniform StructuredBuffer<float> LSE,
    uniform RWStructuredBuffer<float> dQ,
    uniform RWStructuredBuffer<float> dK,
    uniform RWStructuredBuffer<float> dV,
    uniform StructuredBuffer<int> qStrides,
    uniform StructuredBuffer<int> kStrides,
    uniform StructuredBuffer<int> vStrides,
    uniform StructuredBuffer<int> oStrides,
    uniform StructuredBuffer<int> doStrides,
    uniform StructuredBuffer<int> dkStrides,
    uniform StructuredBuffer<int> dvStrides,
    uniform int B,
    uniform int H,
    uniform int L,
    uniform int D,
    uniform float scale,
    uniform int causal
)
{
    int j = dispatchThreadID.x;   // key/value index
    int bh = dispatchThreadID.y;

    if (j >= L || bh >= B * H) return;

    int b = bh / H;
    int h = bh - b * H;

    int kBase = b * kStrides[0] + h * kStrides[1] + j * kStrides[2];
    int vBase = b * vStrides[0] + h * vStrides[1] + j * vStrides[2];
    int dkBase = b * dkStrides[0] + h * dkStrides[1] + j * dkStrides[2];
    int dvBase = b * dvStrides[0] + h * dvStrides[1] + j * dvStrides[2];

    // accumulators initializing
    float dK_acc[FA_HEAD_DIM];
    float dV_acc[FA_HEAD_DIM];
    for (int d = 0; d < D && d < FA_HEAD_DIM; ++d) {
        dK_acc[d] = 0.0f;
        dV_acc[d] = 0.0f;
    }

    // now we iterate over query positions
    for (int i = 0; i < L; ++i) {
        // causal mask: position i can only attend to j iff j <= i
        if (causal != 0 && j > i) continue;

        int qBase = b * qStrides[0] + h * qStrides[1] + i * qStrides[2];
        int oBase = b * oStrides[0] + h * oStrides[1] + i * oStrides[2];
        int doBase = b * doStrides[0] + h * doStrides[1] + i * doStrides[2];
        int lseIdx = b * H * L + h * L + i;

        // recompute attention score: s_ij = Q[i] · K[j] * scale
        float score = 0.0f;
        for (int d = 0; d < D; ++d) {
            score += Q[qBase + d * qStrides[3]] * K[kBase + d * kStrides[3]];
        }
        score *= scale;

        // recompute attention probability: p_ij = exp(s_ij - LSE[i])
        float lse_i = LSE[lseIdx];
        float p_ij = exp(score - lse_i);

        // D_i = sum_d(dO[i,d] * O[i,d])
        float D_i = 0.0f;
        for (int d = 0; d < D; ++d) {
            D_i += dO[doBase + d * doStrides[3]] * O[oBase + d * oStrides[3]];
        }

        // dP_ij = sum_d(dO[i,d] * V[j,d])
        float dP_ij = 0.0f;
        for (int d = 0; d < D; ++d) {
            dP_ij += dO[doBase + d * doStrides[3]] * V[vBase + d * vStrides[3]];
        }

        float dS_ij = p_ij * (dP_ij - D_i) * scale;

        for (int d = 0; d < D && d < FA_HEAD_DIM; ++d) {
            dK_acc[d] += dS_ij * Q[qBase + d * qStrides[3]];
            dV_acc[d] += p_ij * dO[doBase + d * doStrides[3]];
        }
    }

    for (int d = 0; d < D && d < FA_HEAD_DIM; ++d) {
        dK[dkBase + d * dkStrides[3]] = dK_acc[d];
        dV[dvBase + d * dvStrides[3]] = dV_acc[d];
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void flash_attention_backward_dq(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> Q,
    uniform StructuredBuffer<float> K,
    uniform StructuredBuffer<float> V,
    uniform StructuredBuffer<float> O,
    uniform StructuredBuffer<float> dO,
    uniform StructuredBuffer<float> LSE,
    uniform RWStructuredBuffer<float> dQ,
    uniform StructuredBuffer<int> qStrides,
    uniform StructuredBuffer<int> kStrides,
    uniform StructuredBuffer<int> vStrides,
    uniform StructuredBuffer<int> oStrides,
    uniform StructuredBuffer<int> doStrides,
    uniform StructuredBuffer<int> dqStrides,
    uniform int B,
    uniform int H,
    uniform int L,
    uniform int D,
    uniform float scale,
    uniform int causal
)
{
    int i = dispatchThreadID.x;   // query index
    int bh = dispatchThreadID.y;

    if (i >= L || bh >= B * H) return;

    int b = bh / H;
    int h = bh - b * H;

    int qBase = b * qStrides[0] + h * qStrides[1] + i * qStrides[2];
    int oBase = b * oStrides[0] + h * oStrides[1] + i * oStrides[2];
    int doBase = b * doStrides[0] + h * doStrides[1] + i * doStrides[2];
    int dqBase = b * dqStrides[0] + h * dqStrides[1] + i * dqStrides[2];
    int lseIdx = b * H * L + h * L + i;
    float lse_i = LSE[lseIdx];

    // D_i = sum_d(dO[i,d] * O[i,d])
    float D_i = 0.0f;
    for (int d = 0; d < D; ++d) {
        D_i += dO[doBase + d * doStrides[3]] * O[oBase + d * oStrides[3]];
    }

    // init dQ[i] accumulator
    float dQ_acc[FA_HEAD_DIM];
    for (int d = 0; d < D && d < FA_HEAD_DIM; ++d) {
        dQ_acc[d] = 0.0f;
    }

    // now we iterate over key pos
    int j_end = (causal != 0) ? (i + 1) : L;
    for (int j = 0; j < j_end; ++j) {
        int kBase = b * kStrides[0] + h * kStrides[1] + j * kStrides[2];
        int vBase = b * vStrides[0] + h * vStrides[1] + j * vStrides[2];

        // attention score recomputing: s_ij = Q[i] · K[j] * scale
        float score = 0.0f;
        for (int d = 0; d < D; ++d) {
            score += Q[qBase + d * qStrides[3]] * K[kBase + d * kStrides[3]];
        }
        score *= scale;

        // att probability: p_ij = exp(s_ij - LSE[i])
        float p_ij = exp(score - lse_i);

        // dP_ij = sum_d(dO[i,d] * V[j,d])
        float dP_ij = 0.0f;
        for (int d = 0; d < D; ++d) {
            dP_ij += dO[doBase + d * doStrides[3]] * V[vBase + d * vStrides[3]];
        }

        float dS_ij = p_ij * (dP_ij - D_i) * scale;

        // now we accumulate dQ[i] += dS_ij * K[j]
        for (int d = 0; d < D && d < FA_HEAD_DIM; ++d) {
            dQ_acc[d] += dS_ij * K[kBase + d * kStrides[3]];
        }
    }

    for (int d = 0; d < D && d < FA_HEAD_DIM; ++d) {
        dQ[dqBase + d * dqStrides[3]] = dQ_acc[d];
    }
}

groupshared float K_tile[FA_TILE_SIZE * FA_HEAD_DIM];
groupshared float V_tile[FA_TILE_SIZE * FA_HEAD_DIM];

[shader("compute")]
[numthreads(FA_TILE_SIZE, 1, 1)]
void flash_attention_forward_tiled(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uint3 groupThreadID : SV_GroupThreadID,
    uniform StructuredBuffer<float> Q,
    uniform StructuredBuffer<float> K,
    uniform StructuredBuffer<float> V,
    uniform RWStructuredBuffer<float> O,
    uniform RWStructuredBuffer<float> LSE,
    uniform StructuredBuffer<int> qStrides,
    uniform StructuredBuffer<int> kStrides,
    uniform StructuredBuffer<int> vStrides,
    uniform StructuredBuffer<int> oStrides,
    uniform int B,
    uniform int H,
    uniform int L,
    uniform int D,
    uniform float scale,
    uniform int causal
)
{
    int lid = groupThreadID.x;
    int wg_size = FA_TILE_SIZE;
    int i = dispatchThreadID.x;
    int bh = dispatchThreadID.y;

    if (bh >= B * H) return;

    int b = bh / H;
    int h = bh - b * H;

    // this is a safety clamp
    int D_clamped = (D < FA_HEAD_DIM) ? D : FA_HEAD_DIM;

    float m = -1e30f/*-INFINITY*/;
    float s = 0.0f;
    float o_acc[FA_HEAD_DIM];

    if (i < L) {
        for (int d = 0; d < D_clamped; ++d) {
            o_acc[d] = 0.0f;
        }
    }

    int qBase = (i < L) ? (b * qStrides[0] + h * qStrides[1] + i * qStrides[2]) : 0;

    int numTiles = (L + FA_TILE_SIZE - 1) / FA_TILE_SIZE;

    for (int tile = 0; tile < numTiles; ++tile) {
        int tile_start = tile * FA_TILE_SIZE;

        // this is to allow loading of K and V tiles into local memory
        GroupMemoryBarrierWithGroupSync();

        for (int t = lid; t < FA_TILE_SIZE; t += wg_size) {
            int j = tile_start + t;
            if (j < L) {
                int kBase_t = b * kStrides[0] + h * kStrides[1] + j * kStrides[2];
                int vBase_t = b * vStrides[0] + h * vStrides[1] + j * vStrides[2];
                for (int d = 0; d < D_clamped; ++d) {
                    K_tile[t * FA_HEAD_DIM + d] = K[kBase_t + d * kStrides[3]];
                    V_tile[t * FA_HEAD_DIM + d] = V[vBase_t + d * vStrides[3]];
                }
            } else {
                for (int d = 0; d < D_clamped; ++d) {
                    K_tile[t * FA_HEAD_DIM + d] = 0.0f;
                    V_tile[t * FA_HEAD_DIM + d] = 0.0f;
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();

        if (i >= L) continue;

        // tile processing
        for (int t = 0; t < FA_TILE_SIZE; ++t) {
            int j = tile_start + t;
            if (j >= L) break;
            if (causal != 0 && j > i) continue;

            // dot product Q[i] · K[j] computing from local memory
            float dot = 0.0f;
            for (int d = 0; d < D_clamped; ++d) {
                dot += Q[qBase + d * qStrides[3]] * K_tile[t * FA_HEAD_DIM + d];
            }

            float score = dot * scale;
            float new_m = max(m, score);
            float e1 = (m < -1e29f) ? 0.0f : exp(m - new_m);
            float e2 = exp(score - new_m);

            for (int d = 0; d < D_clamped; ++d) {
                o_acc[d] = o_acc[d] * e1 + V_tile[t * FA_HEAD_DIM + d] * e2;
            }

            s = s * e1 + e2;
            m = new_m;
        }
    }

    if (i >= L) return;

    // store LSE and normalize output
    int oBase = b * oStrides[0] + h * oStrides[1] + i * oStrides[2];
    int lseIdx = b * H * L + h * L + i;

    LSE[lseIdx] = m + log(s + 1e-10f);

    float inv_s = (s > 0.0f) ? (1.0f / s) : 0.0f;
    for (int d = 0; d < D_clamped; ++d) {
        O[oBase + d * oStrides[3]] = o_acc[d] * inv_s;
    }
}

groupshared float Q_tile[FA_TILE_SIZE * FA_HEAD_DIM];
groupshared float dO_tile[FA_TILE_SIZE * FA_HEAD_DIM];
groupshared float O_tile[FA_TILE_SIZE * FA_HEAD_DIM];

[shader("compute")]
[numthreads(FA_TILE_SIZE, 1, 1)]
void flash_attention_backward_tiled(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uint3 groupThreadID : SV_GroupThreadID,
    uniform StructuredBuffer<float> Q,
    uniform StructuredBuffer<float> K,
    uniform StructuredBuffer<float> V,
    uniform StructuredBuffer<float> O,
    uniform StructuredBuffer<float> dO,
    uniform StructuredBuffer<float> LSE,
    uniform RWStructuredBuffer<float> dQ,
    uniform RWStructuredBuffer<float> dK,
    uniform RWStructuredBuffer<float> dV,
    uniform StructuredBuffer<int> qStrides,
    uniform StructuredBuffer<int> kStrides,
    uniform StructuredBuffer<int> vStrides,
    uniform StructuredBuffer<int> oStrides,
    uniform StructuredBuffer<int> doStrides,
    uniform StructuredBuffer<int> dkStrides,
    uniform StructuredBuffer<int> dvStrides,
    uniform int B,
    uniform int H,
    uniform int L,
    uniform int D,
    uniform float scale,
    uniform int causal
)
{
    int lid = groupThreadID.x;
    int wg_size = FA_TILE_SIZE;
    int j = dispatchThreadID.x;
    int bh = dispatchThreadID.y;

    if (bh >= B * H) return;

    int b = bh / H;
    int h = bh - b * H;

    // accumulators init
    float dK_acc[FA_HEAD_DIM];
    float dV_acc[FA_HEAD_DIM];

    if (j < L) {
        for (int d = 0; d < D && d < FA_HEAD_DIM; ++d) {
            dK_acc[d] = 0.0f;
            dV_acc[d] = 0.0f;
        }
    }

    int kBase = (j < L) ? (b * kStrides[0] + h * kStrides[1] + j * kStrides[2]) : 0;
    int vBase = (j < L) ? (b * vStrides[0] + h * vStrides[1] + j * vStrides[2]) : 0;
    int dkBase = (j < L) ? (b * dkStrides[0] + h * dkStrides[1] + j * dkStrides[2]) : 0;
    int dvBase = (j < L) ? (b * dvStrides[0] + h * dvStrides[1] + j * dvStrides[2]) : 0;

    int numTiles = (L + FA_TILE_SIZE - 1) / FA_TILE_SIZE;

    for (int tile = 0; tile < numTiles; ++tile) {
        int tile_start = tile * FA_TILE_SIZE;

        GroupMemoryBarrierWithGroupSync();

        for (int t = lid; t < FA_TILE_SIZE; t += wg_size) {
            int i = tile_start + t;
            if (i < L) {
                int qBase_t = b * qStrides[0] + h * qStrides[1] + i * qStrides[2];
                int oBase_t = b * oStrides[0] + h * oStrides[1] + i * oStrides[2];
                int doBase_t = b * doStrides[0] + h * doStrides[1] + i * doStrides[2];
                for (int d = 0; d < D; ++d) {
                    Q_tile[t * D + d] = Q[qBase_t + d * qStrides[3]];
                    dO_tile[t * D + d] = dO[doBase_t + d * doStrides[3]];
                    O_tile[t * D + d] = O[oBase_t + d * oStrides[3]];
                }
            } else {
                for (int d = 0; d < D; ++d) {
                    Q_tile[t * D + d] = 0.0f;
                    dO_tile[t * D + d] = 0.0f;
                    O_tile[t * D + d] = 0.0f;
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();

        if (j >= L) continue;

        // tile processing
        for (int t = 0; t < FA_TILE_SIZE; ++t) {
            int i = tile_start + t;
            if (i >= L) break;
            if (causal != 0 && j > i) continue;

            int lseIdx = b * H * L + h * L + i;
            float lse_i = LSE[lseIdx];

            float score = 0.0f;
            for (int d = 0; d < D; ++d) {
                score += Q_tile[t * D + d] * K[kBase + d * kStrides[3]];
            }
            score *= scale;

            float p_ij = exp(score - lse_i);

            // compute D_i from tiled data
            float D_i = 0.0f;
            for (int d = 0; d < D; ++d) {
                D_i += dO_tile[t * D + d] * O_tile[t * D + d];
            }

            // compute dP_ij
            float dP_ij = 0.0f;
            for (int d = 0; d < D; ++d) {
                dP_ij += dO_tile[t * D + d] * V[vBase + d * vStrides[3]];
            }

            float dS_ij = p_ij * (dP_ij - D_i) * scale;

            // accumulate gradients
            for (int d = 0; d < D && d < FA_HEAD_DIM; ++d) {
                dK_acc[d] += dS_ij * Q_tile[t * D + d];
                dV_acc[d] += p_ij * dO_tile[t * D + d];
            }
        }
    }

    if (j >= L) return;

    for (int d = 0; d < D && d < FA_HEAD_DIM; ++d) {
        dK[dkBase + d * dkStrides[3]] = dK_acc[d];
        dV[dvBase + d * dvStrides[3]] = dV_acc[d];
    }
}

