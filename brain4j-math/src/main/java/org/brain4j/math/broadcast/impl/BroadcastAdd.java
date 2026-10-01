package org.brain4j.math.broadcast.impl;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import org.brain4j.math.broadcast.BaseBroadcast;
import org.brain4j.math.tensor.Tensor;

public class BroadcastAdd extends BaseBroadcast {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    @Override
    protected void sameShape(float[] a, float[] b) {
        for (int i = 0; i < a.length; i++) {
            a[i] += b[i];
        }
    }

    @Override
    protected void sameShapeSimd(float[] a, float[] b) {
        int bound = SPECIES.loopBound(a.length);
        int i = 0;

        for (; i < bound; i += SPECIES.length()) {
            var va = FloatVector.fromArray(SPECIES, a, i);
            var vb = FloatVector.fromArray(SPECIES, b, i);

            va.add(vb).intoArray(a, i);
        }

        for (; i < a.length; i++) {
            a[i] += b[i];
        }
    }

    @Override
    protected void rowWise(float[] a, float[] b, int batch, int dim) {
        for (int r = 0; r < batch; r++) {
            int off = r * dim;

            for (int j = 0; j < dim; j++) {
                a[off + j] += b[j];
            }
        }
    }

    @Override
    protected void rowWiseSimd(float[] a, float[] b, int batch, int dim) {
        int bound = SPECIES.loopBound(dim);

        for (int r = 0; r < batch; r++) {
            int off = r * dim;
            int j = 0;

            for (; j < bound; j += SPECIES.length()) {
                var va = FloatVector.fromArray(SPECIES, a, off + j);
                var vb = FloatVector.fromArray(SPECIES, b, j);

                va.add(vb).intoArray(a, off + j);
            }

            for (; j < dim; j++) {
                a[off + j] += b[j];
            }
        }
    }

    @Override
    protected void bias(Tensor output, Tensor biasTensor) {
        float[] out = output.data();
        float[] bv = biasTensor.data();
        int[] shape = output.shape();

        int batch = shape[0];
        int filters = shape[1];
        int hw = shape[2] * shape[3];
        int strideB = filters * hw;

        for (int b = 0; b < batch; b++) {
            for (int f = 0; f < filters; f++) {
                int base = b * strideB + f * hw;
                float v = bv[f];

                for (int i = 0; i < hw; i++) {
                    out[base + i] += v;
                }
            }
        }
    }

    @Override
    protected void biasSimd(Tensor output, Tensor biasTensor) {
        float[] out = output.data();
        float[] bv = biasTensor.data();
        int[] shape = output.shape();

        int batch = shape[0];
        int filters = shape[1];
        int hw = shape[2] * shape[3];
        int strideB = filters * hw;
        int bound = SPECIES.loopBound(hw);

        for (int b = 0; b < batch; b++) {
            for (int f = 0; f < filters; f++) {
                int base = b * strideB + f * hw;
                var vec = FloatVector.broadcast(SPECIES, bv[f]);
                int i = 0;

                for (; i < bound; i += SPECIES.length()) {
                    var v = FloatVector.fromArray(SPECIES, out, base + i);

                    v.add(vec).intoArray(out, base + i);
                }

                for (; i < hw; i++) {
                    out[base + i] += bv[f];
                }
            }
        }
    }

    @Override
    protected float scalar(float a, float b) {
        return a + b;
    }
}
