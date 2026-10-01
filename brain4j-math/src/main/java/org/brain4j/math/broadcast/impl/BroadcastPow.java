package org.brain4j.math.broadcast.impl;

import org.brain4j.math.broadcast.BaseBroadcast;
import org.brain4j.math.tensor.Tensor;

public class BroadcastPow extends BaseBroadcast {

    @Override
    protected void sameShape(float[] a, float[] b) {
        for (int i = 0; i < a.length; i++) {
            a[i] = (float) Math.pow(a[i], b[i]);
        }
    }

    @Override
    protected void sameShapeSimd(float[] a, float[] b) {
        sameShape(a, b);
    }

    @Override
    protected void rowWise(float[] a, float[] b, int batch, int dim) {
        for (int r = 0; r < batch; r++) {
            int off = r * dim;

            for (int j = 0; j < dim; j++) {
                a[off + j] = (float) Math.pow(a[off + j], b[j]);
            }
        }
    }

    @Override
    protected void rowWiseSimd(float[] a, float[] b, int batch, int dim) {
        rowWise(a, b, batch, dim);
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
                    out[base + i] = (float) Math.pow(out[base + i], v);
                }
            }
        }
    }

    @Override
    protected void biasSimd(Tensor output, Tensor biasTensor) {
        bias(output, biasTensor);
    }

    @Override
    protected float scalar(float a, float b) {
        return (float) Math.pow(a, b);
    }
}
