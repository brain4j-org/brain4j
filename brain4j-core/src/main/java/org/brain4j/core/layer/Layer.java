package org.brain4j.core.layer;

import org.brain4j.core.serializing.BinarySerializable;
import org.brain4j.core.initialization.WeightInitializer;
import org.brain4j.core.loss.LossFunction;
import org.brain4j.core.structure.StatesCache;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.math.Brain4JUtils;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.util.Random;

public abstract class Layer implements BinarySerializable {

    protected static int totalLayers = 0;

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

    public String getLayerName() {
        return this.getClass().getSimpleName();
    }

    public static int getTotalLayers() {
        return totalLayers;
    }

    public boolean canPropagate() {
        return true;
    }

    public boolean canConnect() {
        return true;
    }

    @Override
    public void serialize(DataOutputStream stream) throws Exception {
        stream.writeInt(id);
        stream.writeUTF(activation.getClass().getName());
        stream.writeInt(bias.elements());

        for (int j = 0; j < bias.elements(); j++) {
            stream.writeDouble(bias.get(j));
        }

        boolean hasWeights = weights != null;
        stream.writeBoolean(hasWeights);

        if (hasWeights) {
            weights.serialize(stream);
        }
    }

    @Override
    public void deserialize(DataInputStream stream) throws Exception {
        this.id = stream.readInt();
        this.activation = Brain4JUtils.newInstance(stream.readUTF());
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

    public void compile(
        WeightInitializer weightInit,
        Optimizer optimizer,
        Updater updater
    ) {
        this.weightInit = weightInit;
        this.optimizer = optimizer;
        this.updater = updater;
    }

    public Tensor computeLoss(
        int index,
        StatesCache cache,
        Tensor targets,
        Tensor outputs,
        LossFunction lossFunction
    ) {
        Tensor error = outputs.minus(targets);
        Tensor derivatives = activation.getDerivative(outputs);

        Tensor input = cache.getInputTensor(index);
        Tensor delta = lossFunction.getDelta(error, derivatives);

        Tensor weightsGradient = input.transpose().matmul(delta);
        Tensor biasesGradient = delta.sum(0, false);

        updater.acknowledgeChange(this, weightsGradient, biasesGradient);

        return delta;
    }

    public void connect(
        Random generator,
        Layer previous,
        double bound
    ) {
        if (previous == null) return;

        int input = previous.getTotalNeurons();
        int output = this.getTotalNeurons();

        for (int i = 0; i < bias.elements(); i++) {
            bias.getData()[i] = (float) (2 * generator.nextDouble() - 1);
        }

        this.weights = Tensors.matrix(input, output);

        for (int i = 0; i < weights.elements(); i++) {
            weights.getData()[i] = (float) (2 * generator.nextDouble() - 1);
        }
    }

    public abstract Tensor forward(
        int index,
        StatesCache cache,
        Tensor input,
        boolean training
    );

    public Tensor backward(int index, StatesCache cache, Layer previous, Tensor delta) {
        throw new UnsupportedOperationException("Not implemented for " + this.getClass().getSimpleName());
    }

    public Activation getActivation() {
        return activation;
    }

    public Tensor getWeights() {
        return weights;
    }

    public Tensor getBias() {
        return bias;
    }

    public int getTotalWeights() {
        return weights.elements();
    }

    public int getTotalNeurons() {
        return bias.elements();
    }

    public int getId() {
        return id;
    }
}
