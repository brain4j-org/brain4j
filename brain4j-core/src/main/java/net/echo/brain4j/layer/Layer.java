package net.echo.brain4j.layer;

import net.echo.brain4j.adapters.Adapter;
import net.echo.brain4j.initialization.WeightInitializer;
import net.echo.brain4j.loss.LossFunction;
import net.echo.brain4j.structure.StatesCache;
import net.echo.brain4j.training.optimizer.Optimizer;
import net.echo.brain4j.training.updater.Updater;
import net.echo.math.BrainUtils;
import net.echo.math.activation.Activation;
import net.echo.math.activation.Activations;
import net.echo.math.tensor.Tensor;
import net.echo.math.tensor.Tensors;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.util.Arrays;
import java.util.Random;

public abstract class Layer implements Adapter {

    private static int totalLayers = 0;

    protected WeightInitializer weightInit;
    protected LossFunction lossFunction;
    protected Optimizer optimizer;
    protected Updater updater;
    protected Layer nextLayer;

    protected Activation activation;
    protected Tensor weights;
    protected Tensor bias;
    protected int id;

    public Layer() {
        this(0, Activations.LINEAR);
    }

    public Layer(Activation activation) {
        this(0, activation);
    }

    public Layer(Activations activation) {
        this(0, activation);
    }

    public Layer(int input, Activations activation) {
        this(input, activation.getFunction());
    }

    public Layer(int input, Activation activation) {
        this.id = totalLayers++;
        this.activation = activation;
        this.bias = Tensors.create(input);
        this.weights = Tensors.zeros(0);
    }

    public static int getTotalLayers() {
        return totalLayers;
    }

    public boolean canPropagate() {
        return true;
    }

    public boolean isConvolutional() {
        return false;
    }

    @Override
    public void serialize(DataOutputStream stream) throws Exception {
        stream.writeInt(id);
        stream.writeUTF(activation.getClass().getName());
        stream.writeInt(bias.elements());

        for (int j = 0; j < bias.elements(); j++) {
            stream.writeDouble(bias.get(j));
        }

        boolean hasWeights = weights != null && weights.dimension() == 2;
        stream.writeBoolean(hasWeights);

        if (hasWeights) {
            weights.serialize(stream);
        }
    }

    @Override
    public void deserialize(DataInputStream stream) throws Exception {
        this.id = stream.readInt();
        this.activation = BrainUtils.newInstance(stream.readUTF());
        this.bias = Tensors.zeros(stream.readInt());

        for (int j = 0; j < bias.elements(); j++) {
            bias.set(stream.readDouble(), j);
        }

        boolean hasWeights = stream.readBoolean();
        this.weights = Tensors.zeros(0);

        if (hasWeights) {
            this.weights = weights.deserialize(stream);
        }
    }

    public void compile(WeightInitializer weightInit, LossFunction lossFunction, Optimizer optimizer, Updater updater) {
        this.weightInit = weightInit;
        this.lossFunction = lossFunction;
        this.optimizer = optimizer;
        this.updater = updater;
    }

    public Tensor computeLoss(StatesCache cache, Tensor targets, Tensor outputs, LossFunction lossFunction) {
        Tensor error = outputs.minus(targets);
        Tensor derivatives = activation.getDerivative(outputs);

        Tensor input = cache.getInputTensor(this);
        Tensor delta = lossFunction.getDelta(error, derivatives);

        Tensor weightsGradient = input.transpose().matmul(delta);
        Tensor biasesGradient = delta.sum(0, false);

        updater.acknowledgeChange(this, weightsGradient, biasesGradient);

        return delta;
    }

    public void connect(Random generator, Layer previous, double bound) {
        if (previous == null) return;

        int input = previous.getTotalNeurons();
        int output = this.getTotalNeurons();

        for (int i = 0; i < bias.elements(); i++) {
            bias.set(2 * generator.nextDouble() - 1, i);
        }

        this.weights = Tensors.matrix(input, output);

        for (int i = 0; i < input; i++) {
            for (int j = 0; j < output; j++) {
                double value = generator.nextDouble(2 * bound) - bound;
                this.weights.set(value, i, j);
            }
        }
    }

    public abstract Tensor forward(StatesCache cache, Layer lastLayer, Tensor input, boolean training);

    public Tensor backward(StatesCache cache, Layer previous, Tensor delta) {
        throw new UnsupportedOperationException("Not implemented for " + this.getClass().getSimpleName());
    }

    public Activation getActivation() {
        return activation;
    }

    public Tensor getBias() {
        return bias;
    }

    public Tensor getWeights() {
        return weights;
    }

    public int getTotalParams() {
        return weights.elements();
    }

    public int getTotalNeurons() {
        return bias.elements();
    }

    public int getId() {
        return id;
    }
}
