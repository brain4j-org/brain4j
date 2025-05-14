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
        error.backward();

        Tensor derivatives = activation.getDerivative(outputs);

        Tensor input = cache.input(index); // [batch_size, input_size]
        Tensor delta = lossFunction.getDelta(error, derivatives); // [batch_size, output_size]

        // delta.T = [output_size, batch_size]
        Tensor weightsGradient = delta.transpose().matmul(input); // [output_size, input_size]
        Tensor biasesGradient = delta.sum(0, false);

        updater.change(weightsGradient, biasesGradient, index);

        Tensor newDelta = delta.matmul(weights); // [batch_size, input_size]
        System.out.println("last layer delta = " + newDelta);
        return newDelta; // [batch_size, input_size]
    }

    public Tensor backward(
        Updater updater,
        Optimizer optimizer,
        StatesCache cache,
        Tensor delta,
        int index
    ) {
        Tensor output = cache.output(index); // [batch_size, output_size]
        Tensor input = cache.input(index); // [batch_size, input_size]

        Tensor derivative = activation.getDerivative(output);
        Tensor projectedDelta = delta.mul(derivative); // [batch_size, output_size]

        Tensor manualGradient = projectedDelta.transpose().matmul(input);
        Tensor automaticGradient = weights.grad().transpose(); // [output_size, input_size]

        System.out.println("delta = " + Arrays.toString(delta.shape()));
        System.out.println("input = " + Arrays.toString(input.shape()));
        System.out.println("deriv = " + Arrays.toString(derivative.shape()));
        System.out.println("manua = " + Arrays.toString(manualGradient.shape()));
        System.out.println("autom = " + Arrays.toString(automaticGradient.shape()));

        System.out.println("derivative:");
        System.out.println(derivative);

        System.out.println("input:");
        System.out.println(input);

        System.out.println("manual:");
        System.out.println(manualGradient);

        System.out.println("automatic:");
        System.out.println(automaticGradient);
        Tensor gradBias = projectedDelta.sum(0, false);

        Tensor gradWeights = optimizer.step(index, this, manualGradient);

        clipper.clip(gradWeights);
        clipper.clip(gradBias);

        updater.change(gradWeights, gradBias, index);

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
