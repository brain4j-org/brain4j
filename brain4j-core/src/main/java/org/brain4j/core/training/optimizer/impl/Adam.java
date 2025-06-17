package org.brain4j.core.training.optimizer.impl;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.model.Model;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

import java.util.HashMap;
import java.util.Map;

public class Adam extends Optimizer {

    // Momentum vectors
    protected Map<Layer, Tensor> firstMomentum;
    protected Map<Layer, Tensor> secondMomentum;

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
        this.firstMomentum = new HashMap<>();
        this.secondMomentum = new HashMap<>();
    }

    @Override
    public Tensor step(Layer layer, Tensor gradient) {
        Tensor first = firstMomentum.get(layer);
        Tensor second = secondMomentum.get(layer);

        if (first == null || second == null) {
            first = Tensors.zeros(gradient.shape());
            second = Tensors.zeros(gradient.shape());
        }

        Tensor gradSquared = gradient.times(gradient);

        first.mul(beta1).add(gradient.times(1 - beta1));
        second.mul(beta2).add(gradSquared.mul(1 - beta2));

        firstMomentum.put(layer, first);
        secondMomentum.put(layer, second);

        double biasCorrection1 = 1 - beta1Timestep;
        double biasCorrection2 = 1 - beta2Timestep;

        Tensor mHat = first.divide(biasCorrection1);
        Tensor vHat = second.divide(biasCorrection2);

        return mHat.div(vHat.sqrt().add(epsilon));
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

    public double beta2() {
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