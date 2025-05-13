package org.brain4j.core.layer;

import org.brain4j.core.loss.LossFunction;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.core.training.StatesCache;

import java.util.Random;

public abstract class Layer {

    protected final Activation activation;
    protected Tensor weights;
    protected Tensor bias;

    public Layer(Activation activation) {
        this.activation = activation;
    }

    public abstract void connect(Layer previous);

    public abstract void initWeights(Random generator, double bound);

    public abstract Tensor forward(StatesCache cache, Tensor input, int index, boolean training);

    public abstract int size();

    public Tensor computeLoss(
        Updater updater,
        StatesCache cache,
        Tensor targets,
        Tensor outputs,
        LossFunction lossFunction,
        int index
    ) {
        Tensor error = outputs.minus(targets);
        Tensor derivatives = activation.getDerivative(outputs);

        Tensor input = cache.input(index);
        Tensor delta = lossFunction.getDelta(error, derivatives);

        Tensor weightsGradient = input.transpose().matmul(delta);
        Tensor biasesGradient = delta.sum(0, false);

        updater.change(weightsGradient, biasesGradient, index);

        return delta;
    }

    public Tensor backward(StatesCache cache, Layer last, Tensor delta, int index) {
        return delta; // TODO: Implement this using computational graphs
    }

    public boolean canPropagate() {
        return true;
    }

    public Activation activation() {
        return activation;
    }

    public Tensor weights() {
        return weights;
    }

    public Tensor bias() {
        return bias;
    }
}
