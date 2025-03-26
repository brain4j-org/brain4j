package net.echo.brain4j.training.optimizer.impl;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.training.optimizer.Optimizer;
import net.echo.math4j.math.tensor.Tensor;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

public class GradientDescent extends Optimizer {

    public GradientDescent(double learningRate) {
        super(learningRate);
    }

    @Override
    public Tensor optimize(Layer layer, Tensor delta, Tensor output) {
        return delta.matmul(output).mul(learningRate);
    }

    @Override
    public void serialize(DataOutputStream dataOutputStream) throws IOException {
        dataOutputStream.writeDouble(learningRate);
    }

    @Override
    public void deserialize(DataInputStream dataInputStream) throws IOException {
        this.learningRate = dataInputStream.readDouble();
    }
}
