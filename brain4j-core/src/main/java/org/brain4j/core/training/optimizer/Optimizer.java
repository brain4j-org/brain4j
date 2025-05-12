package org.brain4j.core.training.optimizer;

import org.brain4j.core.serializing.BinarySerializable;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.model.Model;
import org.brain4j.math.tensor.Tensor;

import java.io.DataInputStream;
import java.io.DataOutputStream;

public abstract class Optimizer implements BinarySerializable {

    protected double learningRate;

    protected Optimizer() {
    }

    public Optimizer(double learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public void serialize(DataOutputStream stream) throws Exception {
        stream.writeDouble(learningRate);
    }

    @Override
    public void deserialize(DataInputStream stream) throws Exception {
        this.learningRate = stream.readDouble();
    }

    public abstract Tensor optimize(Layer layer, Tensor delta, Tensor output);

    public void postInitialize(Model model) {
    }

    public void postBatch() {
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
}