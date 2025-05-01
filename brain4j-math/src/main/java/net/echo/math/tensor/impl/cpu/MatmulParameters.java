package net.echo.math.tensor.impl.cpu;

public record MatmulParameters(
        int m,
        int n,
        int p,
        float[] A,
        float[] B,
        float[] C
) {

}
