package net.echo.brain4j.layer;

import com.google.gson.annotations.JsonAdapter;
import net.echo.brain4j.activation.Activation;
import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.adapters.Adapter;
import net.echo.brain4j.adapters.json.LayerAdapter;
import net.echo.brain4j.loss.LossFunction;
import net.echo.brain4j.model.initialization.WeightInitializer;
import net.echo.brain4j.structure.Parameters;
import net.echo.brain4j.structure.StatesCache;
import net.echo.brain4j.training.optimizer.Optimizer;
import net.echo.brain4j.training.updater.Updater;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.util.Random;

@JsonAdapter(LayerAdapter.class)
public abstract class Layer implements Adapter {

    protected WeightInitializer weightInit;
    protected LossFunction lossFunction;
    protected Optimizer optimizer;
    protected Updater updater;
    protected Layer nextLayer;

    protected Tensor weights;
    protected Tensor bias;
    protected Activation activation;
    protected int id;

    public Layer() {
        this.activation = Activations.LINEAR.getFunction();
        this.bias = TensorFactory.zeros(0);
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

    public void preConnect(Layer previous, Layer next) {

    }

    public void connect(Random generator, Layer previous, Layer next, double bound) {
        this.nextLayer = next;

        int nIn = getTotalNeurons();
        int nOut = next.getTotalNeurons();

        int input = bias.elements();
        int output = next.getTotalNeurons();

        this.weights = TensorFactory
                .matrix(output, input)
                .fill(() -> generator.nextDouble(2 * bound) - bound);
    }

    public Tensor forward(StatesCache cache, Layer lastLayer, Tensor input) {
        throw new UnsupportedOperationException("Not implemented for this class.");
    }

    public Tensor propagate(StatesCache cache, Layer previous, Tensor delta) {
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
