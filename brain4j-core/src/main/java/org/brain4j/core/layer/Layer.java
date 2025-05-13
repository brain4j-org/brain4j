package org.brain4j.core.layer;

import org.brain4j.core.clipper.GradientClipper;
import org.brain4j.core.loss.LossFunction;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.core.training.StatesCache;
import org.brain4j.math.tensor.Tensors;

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
        outputs.backward();

        Tensor error = outputs.minus(targets);
        Tensor derivatives = activation.getDerivative(outputs);

        Tensor input = cache.input(index); // [batch_size, input_size]
        Tensor delta = lossFunction.getDelta(error, derivatives); // [batch_size, output_size]

        // delta.T = [output_size, batch_size]
        Tensor weightsGradient = delta.transpose().matmul(input); // [output_size, input_size]
        Tensor biasesGradient = delta.sum(0, false);

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
        Tensor input = cache.input(index); // [batch_size, input_size]
        Tensor deltaThis = input.grad();

        Tensor gradWeights = weights.grad().transpose(); // [output_size, input_size]
        Tensor gradBias = delta.sum(0, false); // [output_size]

        gradWeights = optimizer.step(index, this, gradWeights);

        clipper.clip(gradWeights);
        clipper.clip(gradBias);

        updater.change(gradWeights, gradBias, index);

        return deltaThis;
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
