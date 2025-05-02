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

    private Lion() {
    }

    public Lion(double learningRate, double beta) {
        super(learningRate);
        this.beta = beta;
    }

    @Override
    public void serialize(DataOutputStream stream) throws Exception {
        super.serialize(stream);
        stream.writeDouble(beta);
    }

    @Override
    public void deserialize(DataInputStream stream) throws Exception {
        super.deserialize(stream);
        this.beta = stream.readDouble();
    }

    @Override
    public void postInitialize(Model model) {
        this.momentumHistory = new Tensor[Layer.getTotalLayers()];
    }

    @Override
    public Tensor optimize(Layer layer, Tensor delta, Tensor output) {
        float factor = (float) (1 - beta);

        Tensor gradient = output.transpose().matmul(delta);
        Tensor signGrad = gradient.sign().mul(factor);

        Tensor momentum = momentumHistory[layer.getId()];

        if (momentum == null) {
            momentum = Tensors.create(gradient.shape());
        }

        momentum = momentum.mul(beta).add(signGrad);
        momentumHistory[layer.getId()] = momentum;

        return momentum.sign();
    }

    public double getBeta() {
        return beta;
    }

    public void setBeta(float beta) {
        this.beta = beta;
    }
}
