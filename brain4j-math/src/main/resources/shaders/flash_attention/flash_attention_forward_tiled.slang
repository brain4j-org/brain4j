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
