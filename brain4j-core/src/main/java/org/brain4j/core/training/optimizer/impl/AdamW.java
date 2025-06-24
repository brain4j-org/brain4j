package org.brain4j.core.training.optimizer.impl;

import org.brain4j.math.tensor.Tensor;

public class AdamW extends Adam {

    private double weightDecay;

    private AdamW() {
        this(0);
    }

    public AdamW(double learningRate) {
        this(learningRate, 0.001);
    }

    public AdamW(double learningRate, double weightDecay) {
        this(learningRate, weightDecay, 0.9, 0.999, 1e-8);
    }

    public AdamW(double learningRate, double weightDecay, double beta1, double beta2, double epsilon) {
        super(learningRate, beta1, beta2, epsilon);
        this.weightDecay = weightDecay;
    }

    @Override
    public Tensor step(Tensor weights, Tensor gradient) {
        Tensor adamValue = super.step(weights, gradient);
        Tensor weightDecayTerm = weights.times(weightDecay);

        return adamValue.add(weightDecayTerm);
    }

    public double weightDecay() {
        return weightDecay;
    }

    public void setWeightDecay(double weightDecay) {
        this.weightDecay = weightDecay;
    }
}