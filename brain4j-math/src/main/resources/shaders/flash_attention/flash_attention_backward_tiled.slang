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
