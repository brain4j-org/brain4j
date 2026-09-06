static const int TILE_SIZE = 16;
static const int MAX_KERNEL = 9;
static const int MAX_INPUT_TILE = TILE_SIZE * 4 + MAX_KERNEL - 1;

groupshared float Tile[MAX_INPUT_TILE][MAX_INPUT_TILE + 1];

[shader("compute")]
[numthreads(16, 16, 1)]
void convolve2d_fft_extract(
    uint3 id : SV_DispatchThreadID,
    uniform StructuredBuffer<float> result,
    uniform RWStructuredBuffer<float> output,
    uniform int outputRows,
    uniform int outputCols,
    uniform int startRow,
    uniform int startCol,
    uniform int resultCols
)
{
    int row = id.x;
    int col = id.y;
    if (row >= outputRows || col >= outputCols) return;

    output[row * outputCols + col] = result[(row + startRow) * resultCols + (col + startCol)];
}
