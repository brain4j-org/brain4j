package org.brain4j.math.operations.impl;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.operations.Operation;

/**
 * Applies layer normalization with scale and bias. Use it to record normalized
 * affine transforms in the computation graph.
 *
 * @param epsilon stability term
 */
public record LayerNormOperation(double epsilon) implements Operation {

    /**
     * {@inheritDoc}
     */
    @Override
    public int requiredInputs() {
        return 3;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor compute(Tensor... inputs) {
        // TODO: refactor this
        Tensor normed = inputs[0].copy(); // .layerNorm(epsilon);
        return normed.times(inputs[1]).plus(inputs[2]);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor output, Tensor... inputs) {
        Tensor x = inputs[0];
        Tensor w = inputs[1];

        int features = x.shape()[x.rank() - 1];

        Tensor go = gradOutput.copy();
        Tensor xx = x.copy();

        Tensor mu = xx.sum(-1, true).divide(features);
        Tensor xc = xx.minus(mu);
        Tensor variance = xc.times(xc).sum(-1, true).divide(features);
        Tensor std = variance.add(epsilon).sqrt();
        Tensor xh = xc.divide(std);

        Tensor gradW = AddOperation.reduceTo(go.times(xh), w);
        Tensor gradB = AddOperation.reduceTo(go.copy(), inputs[2]);

        Tensor meanGo = go.sum(-1, true).divide(features);
        Tensor meanGoXh = go.times(xh).sum(-1, true).divide(features);
        Tensor dx = go.minus(meanGo).minus(xh.times(meanGoXh)).times(w).divide(std);

        return new Tensor[] { dx, gradW, gradB };
    }
}
