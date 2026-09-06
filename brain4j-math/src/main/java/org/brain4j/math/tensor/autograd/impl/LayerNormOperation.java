package org.brain4j.math.tensor.autograd.impl;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.autograd.Operation;

public record LayerNormOperation(double epsilon) implements Operation {

    @Override
    public int requiredInputs() {
        return 3;
    }

    @Override
    public Tensor compute(Tensor... inputs) {
        Tensor normed = inputs[0].copy().layerNorm(epsilon);
        return normed.times(inputs[1]).plus(inputs[2]);
    }

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

        // dw = sum(go * x_hat), db = sum(go) over normalized axes
        Tensor gradW = AddOperation.reduceTo(go.times(xh), w);
        Tensor gradB = AddOperation.reduceTo(go.copy(), inputs[2]);

        // dx = (w / std) * (go - mean(go) - x_hat * mean(go * x_hat))
        Tensor meanGo = go.sum(-1, true).divide(features);
        Tensor meanGoXh = go.times(xh).sum(-1, true).divide(features);
        Tensor dx = go.minus(meanGo).minus(xh.times(meanGoXh)).times(w).divide(std);

        return new Tensor[] { dx, gradW, gradB };
    }
}
