package net.echo.brain4j.training.optimizers.impl;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.training.optimizers.Optimizer;
import net.echo.brain4j.training.updater.Updater;

import java.util.List;

public class AdamW extends Optimizer {

    // Momentum vectors
    private ThreadLocal<double[]> firstMomentum;
    private ThreadLocal<double[]> secondMomentum;

    private double beta1Timestep;
    private double beta2Timestep;

    private double beta1;
    private double beta2;
    private double epsilon;
    private double weightDecay;
    private int timestep = 0;

    public AdamW(double learningRate) {
        this(learningRate, 0.01, 0.9, 0.999, 1e-8);
    }

    public AdamW(double learningRate, double weightDecay) {
        this(learningRate, weightDecay, 0.9, 0.999, 1e-8);
    }

    public AdamW(double learningRate, double weightDecay, double beta1, double beta2, double epsilon) {
        super(learningRate);
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.weightDecay = weightDecay;
    }

    @Override
    public void postInitialize() {
        this.firstMomentum = ThreadLocal.withInitial(() -> new double[Synapse.ID_COUNTER]);
        this.secondMomentum = ThreadLocal.withInitial(() -> new double[Synapse.ID_COUNTER]);
    }

    @Override
    public double update(Synapse synapse, Object... params) {
        double[] firstMomentum = (double[]) params[0];
        double[] secondMomentum = (double[]) params[0];

        double gradient = synapse.getOutputNeuron().getDelta() * synapse.getInputNeuron().getValue();

        int synapseId = synapse.getSynapseId();

        double currentFirstMomentum = firstMomentum[synapseId];
        double currentSecondMomentum = secondMomentum[synapseId];

        double m = beta1 * currentFirstMomentum + (1 - beta1) * gradient;
        double v = beta2 * currentSecondMomentum + (1 - beta2) * gradient * gradient;

        firstMomentum[synapseId] = m;
        secondMomentum[synapseId] = v;

        double mHat = m / (1 - beta1Timestep);
        double vHat = v / (1 - beta2Timestep);

        double weightDecayTerm = weightDecay * synapse.getWeight();
        return (learningRate * mHat) / (Math.sqrt(vHat) + epsilon) + weightDecayTerm;
    }

    @Override
    public void postIteration(Updater updater, List<Layer> layers) {
        this.timestep++;

        this.beta1Timestep = Math.pow(beta1, timestep);
        this.beta2Timestep = Math.pow(beta2, timestep);

        double[] firstMomentum = this.firstMomentum.get();
        double[] secondMomentum = this.secondMomentum.get();

        for (Layer layer : layers) {
            for (Synapse synapse : layer.getSynapses()) {
                double change = update(synapse, firstMomentum, secondMomentum);
                updater.acknowledgeChange(synapse, change);
            }
        }
    }

    public double getBeta1() {
        return beta1;
    }

    public void setBeta1(double beta1) {
        this.beta1 = beta1;
    }

    public double getBeta2() {
        return beta2;
    }

    public void setBeta2(double beta2) {
        this.beta2 = beta2;
    }

    public double getEpsilon() {
        return epsilon;
    }

    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }

    public double getWeightDecay() {
        return weightDecay;
    }

    public void setWeightDecay(double weightDecay) {
        this.weightDecay = weightDecay;
    }
}