package org.brain4j.core.layer;

import org.brain4j.core.clipper.GradientClipper;
import org.brain4j.core.loss.LossFunction;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.core.training.StatesCache;

import java.util.Arrays;
import java.util.Random;

public abstract class Layer {

    protected final Activation activation;
    protected final GradientClipper clipper;
    protected Tensor weights;
    protected Tensor bias;

    public Layer(Activation activation, GradientClipper clipper) {
        this.activation = activation;
        this.clipper = clipper;
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
        Tensor error = outputs.subGrad(targets);
        Tensor derivatives = activation.getDerivative(outputs);

        Tensor output = cache.output(index);
        Tensor delta = lossFunction.getDelta(error, derivatives); // [batch_size, output_size]

        output.backward(delta);

        // delta.T = [output_size, batch_size]
        Tensor weightsGradient = weights.grad().transpose();
        Tensor biasesGradient = bias.grad().sum(0, false);

        updater.change(weightsGradient, biasesGradient, index);

        return delta.matmul(weights); // [batch_size, input_size]
    }

    public Tensor backward(
        Updater updater,
        Optimizer optimizer,
        StatesCache cache,
        Tensor delta,
        int index
    ) {
        Tensor weightsGradient = weights.grad().transpose();
        Tensor biasGradient = bias.grad().sum(0, false);

        weightsGradient = optimizer.step(index, this, weightsGradient);

        clipper.clip(weightsGradient);
        clipper.clip(biasGradient);

        updater.change(weightsGradient, biasGradient, index);

        return delta.matmul(weights);
    }

    public boolean skipPropagate() {
        return false;
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
