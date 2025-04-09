package net.echo.brain4j.training.optimizer.impl;

import net.echo.brain4j.layer.Layer;
import net.echo.math4j.math.tensor.Tensor;

import java.io.DataInputStream;
import java.io.DataOutputStream;

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
    public void serialize(DataOutputStream dataOutputStream) throws Exception {
        super.serialize(dataOutputStream);
        dataOutputStream.writeDouble(weightDecay);
    }

    @Override
    public void deserialize(DataInputStream dataInputStream) throws Exception {
        super.deserialize(dataInputStream);
        this.weightDecay = dataInputStream.readDouble();
    }

    @Override
    public Tensor optimize(Layer layer, Tensor delta, Tensor output) {
        Tensor adamValue = super.optimize(layer, delta, output);
        Tensor weightDecayTerm = layer.getWeights().clone().mul(weightDecay);

        return adamValue.add(weightDecayTerm);
    }

    public double getWeightDecay() {
        return weightDecay;
    }

    public void setWeightDecay(double weightDecay) {
        this.weightDecay = weightDecay;
    }
}