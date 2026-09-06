package org.brain4j.math.tensor.autograd.impl;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.autograd.Operation;

public record RMSNormOperation(double epsilon) implements Operation {

    @Override
    public Tensor compute(Tensor... inputs) {
        Tensor rms = rmsOf(inputs[0].copy());
        return inputs[0].copy().divide(rms).times(inputs[1]);
    }

    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor output, Tensor... inputs) {
        Tensor x = inputs[0];
        Tensor w = inputs[1];

        // Detached working copies.
        Tensor go = gradOutput.copy();
        Tensor rms = rmsOf(x.copy());
        Tensor xh = x.copy().divide(rms);

        // dw = sum(go * x_hat) over normalized axes
        Tensor gradW = AddOperation.reduceTo(go.times(xh), w);

        // dx = (go - x_hat * mean(go * x_hat)) * w / rms
        int features = x.shape()[x.rank() - 1];
        Tensor meanGoXh = go.times(xh).sum(-1, true).divide(features);
        Tensor dx = go.minus(xh.times(meanGoXh)).times(w).divide(rms);

        return new Tensor[] { dx, gradW };
    }

    private Tensor rmsOf(Tensor owned) {
        int features = owned.shape()[owned.rank() - 1];
        Tensor meanSq = owned.times(owned).sum(-1, true).divide(features);
        return meanSq.add(epsilon).sqrt();
    }
}
