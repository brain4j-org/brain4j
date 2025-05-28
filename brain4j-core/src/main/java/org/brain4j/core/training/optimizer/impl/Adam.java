package org.brain4j.core.training.optimizer.impl;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.model.Model;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

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
    public void initialize(Model model) {
        this.beta1Timestep = Math.pow(beta1, timestep);
        this.beta2Timestep = Math.pow(beta2, timestep);

        this.firstMomentum = new Tensor[model.size()];
        this.secondMomentum = new Tensor[model.size()];
    }

    @Override
    public Tensor step(int index, Layer layer, Tensor gradient) {
        Tensor firstMomentum = this.firstMomentum[index];
        Tensor secondMomentum = this.secondMomentum[index];

        if (firstMomentum == null || secondMomentum == null) {
            firstMomentum = Tensors.zeros(gradient.shape());
            secondMomentum = Tensors.zeros(gradient.shape());
        }

        Tensor gradSquared = gradient.times(gradient);

        firstMomentum = firstMomentum.mul(beta1).add(gradient.times(1 - beta1));
        secondMomentum = secondMomentum.mul(beta2).add(gradSquared.mul(1 - beta2));

        this.firstMomentum[index] = firstMomentum;
        this.secondMomentum[index] = secondMomentum;

        double biasCorrection1 = 1 - beta1Timestep;
        double biasCorrection2 = 1 - beta2Timestep;

        Tensor mHat = firstMomentum.divide(biasCorrection1);
        Tensor vHat = secondMomentum.divide(biasCorrection2);

        // transpose of 2.9
        return mHat.div(vHat.sqrt().add(epsilon)).transpose();
    }

    @Override
    public void postBatch() {
        this.timestep++;
        this.beta1Timestep *= beta1;
        this.beta2Timestep *= beta2;
    }

    public double beta1() {
        return beta1;
    }

    public void setBeta1(double beta1) {
        this.beta1 = (float) beta1;
    }

    public double beata2() {
        return beta2;
    }

    public void setBeta2(double beta2) {
        this.beta2 = (float) beta2;
    }

    public double epsilon() {
        return epsilon;
    }

    public void setEpsilon(double epsilon) {
        this.epsilon = (float) epsilon;
    }
}