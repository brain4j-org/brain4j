package org.brain4j.core.training.optimizer;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.model.Model;
import org.brain4j.math.tensor.Tensor;

public abstract class Optimizer {

    protected double learningRate;

    protected Optimizer() {
    }

    public Optimizer(double learningRate) {
        this.learningRate = learningRate;
    }

    public abstract Tensor step(int index, Layer layer, Tensor gradient);

    public void initialize(Model model) {
    }

    public void postBatch() {
    }

    public double learningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
}