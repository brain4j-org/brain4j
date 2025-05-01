package org.brain4j.core.training.optimizer.impl;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.model.Model;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

import java.io.DataInputStream;
import java.io.DataOutputStream;

public class Adam extends Optimizer {

    // Momentum vectors
    protected Tensor[] firstMomentum;
    protected Tensor[] secondMomentum;

    protected double beta1Timestep;
    protected double beta2Timestep;

    protected float beta1;
    protected float beta2;
    protected float epsilon;

    protected int timestep = 1;

    private Adam() {
    }

    public Adam(double learningRate) {
        this(learningRate, 0.9, 0.999, 1e-8);
    }

    public Adam(double learningRate, double beta1, double beta2, double epsilon) {
        super(learningRate);
        this.beta1 = (float) beta1;
        this.beta2 = (float) beta2;
        this.epsilon = (float) epsilon;
    }

    @Override
    public void postInitialize(Model model) {
        this.beta1Timestep = Math.pow(beta1, timestep);
        this.beta2Timestep = Math.pow(beta2, timestep);

        this.firstMomentum = new Tensor[model.getLayers().size()];
        this.secondMomentum = new Tensor[model.getLayers().size()];
    }

    @Override
    public void serialize(DataOutputStream stream) throws Exception {
        super.serialize(stream);
        stream.writeFloat(timestep);
        stream.writeFloat(beta1);
        stream.writeFloat(beta2);
        stream.writeFloat(epsilon);
    }

    @Override
    public void deserialize(DataInputStream dataInputStream) throws Exception {
        super.deserialize(dataInputStream);
        this.timestep = dataInputStream.readInt();
        this.beta1 = dataInputStream.readFloat();
        this.beta2 = dataInputStream.readFloat();
        this.epsilon = dataInputStream.readFloat();
        this.beta1Timestep = Math.pow(beta1, timestep);
        this.beta2Timestep = Math.pow(beta2, timestep);
    }

    @Override
    public Tensor optimize(Layer layer, Tensor delta, Tensor output) {
        // [n_out, batch_size] x [batch_size, n_out]
        Tensor gradient = output.transpose().matmul(delta);

        Tensor firstMomentum = this.firstMomentum[layer.getId()];
        Tensor secondMomentum = this.secondMomentum[layer.getId()];

        if (firstMomentum == null || secondMomentum == null) {
            firstMomentum = Tensors.create(gradient.shape());
            secondMomentum = Tensors.create(gradient.shape());
        }

        Tensor gradSquared = gradient.clone().mul(gradient);

        firstMomentum = firstMomentum.mul(beta1).add(gradient.mul(1 - beta1));
        secondMomentum = secondMomentum.mul(beta2).add(gradSquared.mul(1 - beta2));

        this.firstMomentum[layer.getId()] = firstMomentum;
        this.secondMomentum[layer.getId()] = secondMomentum;

        double biasCorrection1 = 1 - beta1Timestep;
        double biasCorrection2 = 1 - beta2Timestep;

        Tensor mHat = firstMomentum.clone().div(biasCorrection1);
        Tensor vHat = secondMomentum.clone().div(biasCorrection2);

        return mHat.div(vHat.sqrt().add(epsilon));
    }

    @Override
    public void postBatch() {
        this.timestep++;
        this.beta1Timestep *= beta1;
        this.beta2Timestep *= beta2;
    }

    public double getBeta1() {
        return beta1;
    }

    public void setBeta1(double beta1) {
        this.beta1 = (float) beta1;
    }

    public double getBeta2() {
        return beta2;
    }

    public void setBeta2(double beta2) {
        this.beta2 = (float) beta2;
    }

    public double getEpsilon() {
        return epsilon;
    }

    public void setEpsilon(double epsilon) {
        this.epsilon = (float) epsilon;
    }
}