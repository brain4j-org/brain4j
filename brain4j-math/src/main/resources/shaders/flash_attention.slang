static const int FA_TILE_SIZE = 16;
static const int FA_HEAD_DIM = 64;

groupshared float K_tile[FA_TILE_SIZE * FA_HEAD_DIM];
groupshared float V_tile[FA_TILE_SIZE * FA_HEAD_DIM];

int offset4(StructuredBuffer<int> strides, int b, int h, int i, int d)
{
    return b * strides[0] + h * strides[1] + i * strides[2] + d * strides[3];
}

float qk_score(
    StructuredBuffer<float> Q,
    StructuredBuffer<float> K,
    StructuredBuffer<int> qStrides,
    StructuredBuffer<int> kStrides,
    int b,
    int h,
    int qi,
    int kj,
    int headDim,
    float scale
)
{
    float score = 0.0f;
    for (int d = 0; d < headDim; d++) {
        score += Q[offset4(qStrides, b, h, qi, d)] * K[offset4(kStrides, b, h, kj, d)];
    }
    return score * scale;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void flash_attention_forward(
    uint3 id : SV_DispatchThreadID,
    uniform StructuredBuffer<float> Q,
    uniform StructuredBuffer<float> K,
    uniform StructuredBuffer<float> V,
    uniform RWStructuredBuffer<float> O,
    uniform StructuredBuffer<int> qStrides,
    uniform StructuredBuffer<int> kStrides,
    uniform StructuredBuffer<int> vStrides,
    uniform StructuredBuffer<int> oStrides,
    uniform StructuredBuffer<int> doStrides,
    uniform StructuredBuffer<int> dkStrides,
    uniform StructuredBuffer<int> dvStrides,
    uniform int batchSize,
    uniform int numHeads,
    uniform int seqLen,
    uniform int headDim,
    uniform float scale,
    uniform int causal
)
{
    int i = id.x;
    int h = id.y;
    int b = id.z;
    if (i >= seqLen || h >= numHeads || b >= batchSize) return;

    float m = -3.402823e38f;
    float s = 0.0f;

    for (int d = 0; d < headDim; d++) {
        O[offset4(oStrides, b, h, i, d)] = 0.0f;
    }

    for (int j = 0; j < seqLen; j++) {
        if (causal != 0 && j > i) continue;

        float score = qk_score(Q, K, qStrides, kStrides, b, h, i, j, headDim, scale);
        float newM = max(m, score);
        float alpha = exp(m - newM);
        float p = exp(score - newM);
        float newS = s * alpha + p;

        for (int d = 0; d < headDim; d++) {
            int oIdx = offset4(oStrides, b, h, i, d);
            int vIdx = offset4(vStrides, b, h, j, d);
            O[oIdx] = O[oIdx] * alpha + V[vIdx] * p;
        }

        m = newM;
        s = newS;
    }

    float invS = 1.0f / max(s, 1e-10f);
    for (int d = 0; d < headDim; d++) {
        int oIdx = offset4(oStrides, b, h, i, d);
        O[oIdx] *= invS;
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void flash_attention_forward_with_lse(
    uint3 id : SV_DispatchThreadID,
    uniform StructuredBuffer<float> Q,
    uniform StructuredBuffer<float> K,
    uniform StructuredBuffer<float> V,
    uniform RWStructuredBuffer<float> O,
    uniform RWStructuredBuffer<float> LSE,
    uniform StructuredBuffer<int> qStrides,
    uniform StructuredBuffer<int> kStrides,
    uniform StructuredBuffer<int> vStrides,
    uniform StructuredBuffer<int> oStrides,
    uniform StructuredBuffer<int> doStrides,
    uniform StructuredBuffer<int> dkStrides,
    uniform StructuredBuffer<int> dvStrides,
    uniform int batchSize,
    uniform int numHeads,
    uniform int seqLen,
    uniform int headDim,
    uniform float scale,
    uniform int causal
)
{
    int i = id.x;
    int h = id.y;
    int b = id.z;
    if (i >= seqLen || h >= numHeads || b >= batchSize) return;

    float m = -3.402823e38f;
    float s = 0.0f;

    for (int d = 0; d < headDim; d++) {
        O[offset4(oStrides, b, h, i, d)] = 0.0f;
    }

    for (int j = 0; j < seqLen; j++) {
        if (causal != 0 && j > i) continue;

        float score = qk_score(Q, K, qStrides, kStrides, b, h, i, j, headDim, scale);
        float newM = max(m, score);
        float alpha = exp(m - newM);
        float p = exp(score - newM);
        float newS = s * alpha + p;

        for (int d = 0; d < headDim; d++) {
            int oIdx = offset4(oStrides, b, h, i, d);
            int vIdx = offset4(vStrides, b, h, j, d);
            O[oIdx] = O[oIdx] * alpha + V[vIdx] * p;
        }

        m = newM;
        s = newS;
    }

    float invS = 1.0f / max(s, 1e-10f);
    for (int d = 0; d < headDim; d++) {
        int oIdx = offset4(oStrides, b, h, i, d);
        O[oIdx] *= invS;
    }

    LSE[(b * numHeads + h) * seqLen + i] = m + log(s + 1e-10f);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void flash_attention_backward(
    uint3 id : SV_DispatchThreadID,
    uniform StructuredBuffer<float> Q,
    uniform StructuredBuffer<float> K,
    uniform StructuredBuffer<float> V,
    uniform StructuredBuffer<float> O,
    uniform StructuredBuffer<float> dO,
    uniform StructuredBuffer<float> LSE,
    uniform RWStructuredBuffer<float> dK,
    uniform RWStructuredBuffer<float> dV,
    uniform StructuredBuffer<int> qStrides,
    uniform StructuredBuffer<int> kStrides,
    uniform StructuredBuffer<int> vStrides,
    uniform StructuredBuffer<int> oStrides,
    uniform StructuredBuffer<int> doStrides,
    uniform StructuredBuffer<int> dqStrides,
    uniform StructuredBuffer<int> dkStrides,
    uniform StructuredBuffer<int> dvStrides,
    uniform int batchSize,
    uniform int numHeads,
    uniform int seqLen,
    uniform int headDim,
    uniform float scale,
    uniform int causal
)
{
    int j = id.x;
    int h = id.y;
    int b = id.z;
    if (j >= seqLen || h >= numHeads || b >= batchSize) return;

    for (int d = 0; d < headDim; d++) {
        float accK = 0.0f;
        float accV = 0.0f;

        for (int i = 0; i < seqLen; i++) {
            if (causal != 0 && j > i) continue;

            float score = qk_score(Q, K, qStrides, kStrides, b, h, i, j, headDim, scale);
            float lse = LSE[(b * numHeads + h) * seqLen + i];
            float p = exp(score - lse);

            float doDotV = 0.0f;
            float doDotO = 0.0f;
            for (int x = 0; x < headDim; x++) {
                float doVal = dO[offset4(doStrides, b, h, i, x)];
                doDotV += doVal * V[offset4(vStrides, b, h, j, x)];
                doDotO += doVal * O[offset4(oStrides, b, h, i, x)];
            }

            accV += p * dO[offset4(doStrides, b, h, i, d)];
            accK += p * (doDotV - doDotO) * scale * Q[offset4(qStrides, b, h, i, d)];
        }

        dV[offset4(dvStrides, b, h, j, d)] = accV;
        dK[offset4(dkStrides, b, h, j, d)] = accK;
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void flash_attention_backward_dq(
    uint3 id : SV_DispatchThreadID,
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
    uniform int batchSize,
    uniform int numHeads,
    uniform int seqLen,
    uniform int headDim,
    uniform float scale,
    uniform int causal
)
{
    int i = id.x;
    int h = id.y;
    int b = id.z;
    if (i >= seqLen || h >= numHeads || b >= batchSize) return;

    for (int d = 0; d < headDim; d++) {
        float accQ = 0.0f;

        for (int j = 0; j < seqLen; j++) {
            if (causal != 0 && j > i) continue;

            float score = qk_score(Q, K, qStrides, kStrides, b, h, i, j, headDim, scale);
            float lse = LSE[(b * numHeads + h) * seqLen + i];
            float p = exp(score - lse);

            float doDotV = 0.0f;
            float doDotO = 0.0f;
            for (int x = 0; x < headDim; x++) {
                float doVal = dO[offset4(doStrides, b, h, i, x)];
                doDotV += doVal * V[offset4(vStrides, b, h, j, x)];
                doDotO += doVal * O[offset4(oStrides, b, h, i, x)];
            }

            accQ += p * (doDotV - doDotO) * scale * K[offset4(kStrides, b, h, j, d)];
        }

        dQ[offset4(dqStrides, b, h, i, d)] = accQ;
    }
}

[shader("compute")]
[numthreads(FA_TILE_SIZE, 1, 1)]
void flash_attention_forward_tiled(
    uint3 groupID : SV_GroupID,
    uint3 localID : SV_GroupThreadID,
    uniform StructuredBuffer<float> Q,
    uniform StructuredBuffer<float> K,
    uniform StructuredBuffer<float> V,
    uniform RWStructuredBuffer<float> O,
    uniform RWStructuredBuffer<float> LSE,
    uniform StructuredBuffer<int> qStrides,
    uniform StructuredBuffer<int> kStrides,
    uniform StructuredBuffer<int> vStrides,
    uniform StructuredBuffer<int> oStrides,
    uniform StructuredBuffer<int> doStrides,
    uniform StructuredBuffer<int> dqStrides,
    uniform StructuredBuffer<int> dkStrides,
    uniform StructuredBuffer<int> dvStrides,
    uniform int batchSize,
    uniform int numHeads,
    uniform int seqLen,
    uniform int headDim,
    uniform float scale,
    uniform int causal
)
{
    int local = localID.x;
    int i = groupID.x * FA_TILE_SIZE + local;
    int h = groupID.y;
    int b = groupID.z;
    if (h >= numHeads || b >= batchSize || headDim > FA_HEAD_DIM) return;

    float m = -3.402823e38f;
    float s = 0.0f;
    float acc[FA_HEAD_DIM];

    for (int d = 0; d < FA_HEAD_DIM; d++) {
        acc[d] = 0.0f;
    }

    for (int tileStart = 0; tileStart < seqLen; tileStart += FA_TILE_SIZE) {
        int keyIndex = tileStart + local;

        if (keyIndex < seqLen) {
            for (int d = 0; d < headDim; d++) {
                K_tile[local * FA_HEAD_DIM + d] = K[offset4(kStrides, b, h, keyIndex, d)];
                V_tile[local * FA_HEAD_DIM + d] = V[offset4(vStrides, b, h, keyIndex, d)];
            }
        } else {
            for (int d = 0; d < headDim; d++) {
                K_tile[local * FA_HEAD_DIM + d] = 0.0f;
                V_tile[local * FA_HEAD_DIM + d] = 0.0f;
            }
        }

        GroupMemoryBarrierWithGroupSync();

        if (i < seqLen) {
            for (int tj = 0; tj < FA_TILE_SIZE; tj++) {
                int j = tileStart + tj;
                if (j >= seqLen) continue;
                if (causal != 0 && j > i) continue;

                float score = 0.0f;
                for (int d = 0; d < headDim; d++) {
                    score += Q[offset4(qStrides, b, h, i, d)] * K_tile[tj * FA_HEAD_DIM + d];
                }
                score *= scale;

                float newM = max(m, score);
                float alpha = exp(m - newM);
                float p = exp(score - newM);
                float newS = s * alpha + p;

                for (int d = 0; d < headDim; d++) {
                    acc[d] = acc[d] * alpha + V_tile[tj * FA_HEAD_DIM + d] * p;
                }

                m = newM;
                s = newS;
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (i < seqLen) {
        float invS = 1.0f / max(s, 1e-10f);
        for (int d = 0; d < headDim; d++) {
            O[offset4(oStrides, b, h, i, d)] = acc[d] * invS;
        }
        LSE[(b * numHeads + h) * seqLen + i] = m + log(s + 1e-10f);
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void flash_attention_backward_tiled(
    uint3 id : SV_DispatchThreadID,
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
    uniform StructuredBuffer<int> dqStrides,
    uniform StructuredBuffer<int> dkStrides,
    uniform StructuredBuffer<int> dvStrides,
    uniform int batchSize,
    uniform int numHeads,
    uniform int seqLen,
    uniform int headDim,
    uniform float scale,
    uniform int causal
)
{
    int idx = id.x;
    int h = id.y;
    int b = id.z;
    if (idx >= seqLen || h >= numHeads || b >= batchSize) return;

    for (int d = 0; d < headDim; d++) {
        float accQ = 0.0f;
        for (int j = 0; j < seqLen; j++) {
            if (causal != 0 && j > idx) continue;

            float score = qk_score(Q, K, qStrides, kStrides, b, h, idx, j, headDim, scale);
            float lse = LSE[(b * numHeads + h) * seqLen + idx];
            float p = exp(score - lse);

            float doDotV = 0.0f;
            float doDotO = 0.0f;
            for (int x = 0; x < headDim; x++) {
                float doVal = dO[offset4(doStrides, b, h, idx, x)];
                doDotV += doVal * V[offset4(vStrides, b, h, j, x)];
                doDotO += doVal * O[offset4(oStrides, b, h, idx, x)];
            }

            float ds = p * (doDotV - doDotO);
            accQ += ds * scale * K[offset4(kStrides, b, h, j, d)];
        }
        dQ[offset4(dqStrides, b, h, idx, d)] = accQ;
    }

    for (int d = 0; d < headDim; d++) {
        float accK = 0.0f;
        float accV = 0.0f;

        for (int i = 0; i < seqLen; i++) {
            if (causal != 0 && idx > i) continue;

            float score = qk_score(Q, K, qStrides, kStrides, b, h, i, idx, headDim, scale);
            float lse = LSE[(b * numHeads + h) * seqLen + i];
            float p = exp(score - lse);

            float doDotV = 0.0f;
            float doDotO = 0.0f;
            for (int x = 0; x < headDim; x++) {
                float doVal = dO[offset4(doStrides, b, h, i, x)];
                doDotV += doVal * V[offset4(vStrides, b, h, idx, x)];
                doDotO += doVal * O[offset4(oStrides, b, h, i, x)];
            }

            accV += p * dO[offset4(doStrides, b, h, i, d)];
            accK += p * (doDotV - doDotO) * scale * Q[offset4(qStrides, b, h, i, d)];
        }

        dK[offset4(dkStrides, b, h, idx, d)] = accK;
        dV[offset4(dvStrides, b, h, idx, d)] = accV;
    }
}
