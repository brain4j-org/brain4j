package org.brain4j.math.tensor.impl.cpu.matmul;

public record MatmulParameters(
        int m,
        int n,
        int p,
        float[] A,
        float[] B,
        float[] C,
        int np,
        int mn,
        int mp
) {

}
