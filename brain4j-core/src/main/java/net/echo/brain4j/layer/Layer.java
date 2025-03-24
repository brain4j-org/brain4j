package net.echo.brain4j.layer;

import com.google.gson.annotations.JsonAdapter;
import net.echo.brain4j.activation.Activation;
import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.adapters.Adapter;
import net.echo.brain4j.adapters.json.LayerAdapter;
import net.echo.brain4j.loss.LossFunction;
import net.echo.brain4j.model.initialization.WeightInitializer;
import net.echo.brain4j.structure.cache.Parameters;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.brain4j.training.optimizers.Optimizer;
import net.echo.brain4j.training.updater.Updater;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.util.Random;

@JsonAdapter(LayerAdapter.class)
public abstract class Layer<I, O> implements Adapter {

    protected WeightInitializer weightInit;
    protected LossFunction lossFunction;
    protected Optimizer optimizer;
    protected Updater updater;
    protected Layer<?, ?> nextLayer;

    protected Tensor weights;
    protected Tensor bias;
    protected Activation activation;
    protected int id;

    public Layer() {
        this(Activations.LINEAR);
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
        this.id = Parameters.TOTAL_LAYERS++;
        this.activation = activation;
        this.bias = TensorFactory.create(input);
        this.weights = TensorFactory.zeros(0);
    }

    @Override
    public void serialize(DataOutputStream stream) throws Exception {
        stream.writeInt(id);
        stream.writeInt(bias.elements());
        stream.writeUTF(activation.getClass().getName());
    }

    @Override
    public void deserialize(DataInputStream stream) throws Exception {
        this.id = stream.readInt();

        int totalNeurons = stream.readInt();
        this.bias = TensorFactory.create(totalNeurons);

        String activationClassPath = stream.readUTF();
        Class<?> activationClass = Class.forName(activationClassPath);

        this.activation = (Activation) activationClass.getDeclaredConstructor().newInstance();
    }

    public boolean canPropagate() {
        return true;
    }

    public boolean isConvolutional() {
        return false;
    }

    public void compile(WeightInitializer weightInit, LossFunction lossFunction, Optimizer optimizer, Updater updater) {
        this.weightInit = weightInit;
        this.lossFunction = lossFunction;
        this.optimizer = optimizer;
        this.updater = updater;
    }

    public void init(Random generator) {
        for (int i = 0; i < bias.elements(); i++) {
            double value = 2 * generator.nextDouble() - 1;
            bias.set(value, i);
        }
    }

    public Tensor computeLoss(StatesCache cache, Tensor targets, Tensor outputs, LossFunction lossFunction) {
        Tensor derivative = activation.getDerivative(outputs);

        Tensor error = outputs.clone().sub(targets);

        return error
                .clone()
                .mapWithIndex((i, x) -> lossFunction.getDelta(x, derivative.get(i)));
    }

    public void connect(Random generator, Layer<?, ?> nextLayer, double bound) {
        this.nextLayer = nextLayer;

        int input = bias.elements();
        int output = nextLayer.getTotalNeurons();

        Tensor weights = TensorFactory.matrix(output, input);

        for (int i = 0; i < bias.elements(); i++) {
            for (int j = 0; j < nextLayer.getTotalNeurons(); j++) {
                double value = (generator.nextDouble() * 2 * bound) - bound;
                weights.set(value, j, i);
            }
        }

        this.weights = weights;
    }

    public O forward(StatesCache cache, Layer<?, ?> lastLayer, I input) {
        throw new UnsupportedOperationException("Not implemented for this class.");
    }

    public Tensor propagate(StatesCache cache, Layer<?, ?> previous, Tensor delta) {
        throw new UnsupportedOperationException("Not implemented for this class.");
    }

    public Activation getActivation() {
        return activation;
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

    public Tensor getBias() {
        return bias;
    }

    public Tensor getWeights() {
        return weights;
    }
}
