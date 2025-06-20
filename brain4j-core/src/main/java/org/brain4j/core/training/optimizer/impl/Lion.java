package org.brain4j.core.training.optimizer.impl;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.model.Model;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

import java.util.HashMap;
import java.util.Map;

public class Lion extends Optimizer {

    private Map<Tensor, Tensor> momentumHistory;
    private double beta;

    public Lion(double learningRate, double beta) {
        super(learningRate);
        this.beta = beta;
    }

    @Override
    public Tensor step(Tensor weights, Tensor gradient) {
        float factor = (float) (1 - beta);

        Tensor signGrad = gradient.sign().mul(factor);
        Tensor momentum = momentumHistory.get(weights);

        if (momentum == null) {
            momentum = Tensors.zeros(gradient.shape());
        }

        momentum.mul(beta).add(signGrad);
        momentumHistory.put(weights, momentum);

        return momentum.sign();
    }

    @Override
    public void initialize(Model model) {
        this.momentumHistory = new HashMap<>();
    }

    public double beta() {
        return beta;
    }

    public void setBeta(float beta) {
        this.beta = beta;
    }
}