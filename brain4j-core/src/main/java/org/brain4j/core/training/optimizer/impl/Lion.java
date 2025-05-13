package org.brain4j.core.training.optimizer.impl;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.model.Model;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

import java.io.DataInputStream;
import java.io.DataOutputStream;

public class Lion extends Optimizer {

    private Tensor[] momentumHistory;
    private double beta;

    public Lion(double learningRate, double beta) {
        super(learningRate);
        this.beta = beta;
    }

    @Override
    public Tensor step(int index, Layer layer, Tensor gradient) {
        float factor = (float) (1 - beta);

        Tensor signGrad = gradient.sign().mul(factor);
        Tensor momentum = momentumHistory[index];

        if (momentum == null) {
            momentum = Tensors.create(gradient.shape());
        }

        momentum = momentum.mul(beta).add(signGrad);
        momentumHistory[index] = momentum;

        return momentum.sign();
    }

    @Override
    public void postInitialize(Model model) {
        this.momentumHistory = new Tensor[model.size()];
    }

    public double beta() {
        return beta;
    }

    public void setBeta(float beta) {
        this.beta = beta;
    }
}