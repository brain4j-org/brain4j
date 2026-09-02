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
