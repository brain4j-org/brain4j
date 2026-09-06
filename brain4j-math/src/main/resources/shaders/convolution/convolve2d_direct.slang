static const int TILE_SIZE = 16;
static const int MAX_KERNEL = 9;
static const int MAX_INPUT_TILE = TILE_SIZE * 4 + MAX_KERNEL - 1;

groupshared float Tile[MAX_INPUT_TILE][MAX_INPUT_TILE + 1];

[shader("compute")]
[numthreads(16, 16, 1)]
void convolve2d_direct(
    uint3 id : SV_DispatchThreadID,
    uniform StructuredBuffer<float> input,
    uniform StructuredBuffer<float> kernel,
    uniform RWStructuredBuffer<float> output,
    uniform int inputRows,
    uniform int inputCols,
    uniform int kernelRows,
    uniform int kernelCols,
    uniform int outputRows,
    uniform int outputCols,
    uniform int paddingTop,
    uniform int paddingLeft
)
{
    int outRow = id.x;
    int outCol = id.y;
    if (outRow >= outputRows || outCol >= outputCols) return;

    float sum = 0.0f;
    for (int kr = 0; kr < kernelRows; kr++) {
        int inRow = outRow + kr - paddingTop;
        if (inRow < 0 || inRow >= inputRows) continue;

        for (int kc = 0; kc < kernelCols; kc++) {
            int inCol = outCol + kc - paddingLeft;
            if (inCol < 0 || inCol >= inputCols) continue;

            sum += input[inRow * inputCols + inCol] * kernel[kr * kernelCols + kc];
        }
    }

    output[outRow * outputCols + outCol] = sum;
}
