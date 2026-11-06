package org.brain4j.core.layer.impl;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.commons.Range;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

/**
 * Runs an LSTM over time steps with gated memory.
 * Use it for sequence modeling when long-range context matters.
 */
public class LSTMLayer extends Layer {

    /**
     * Holds the hidden size and sequence output flag.
     */
    public record Config(int hiddenDimension, boolean returnSequences) {}

    protected final Config config;

    /**
     * Create a layer with the given hidden size and output mode.
     */
    public LSTMLayer(int hiddenDimension, boolean returnSequences) {
        this(new Config(hiddenDimension, returnSequences));
    }

    /**
     * Create a layer with the given config.
     */
    public LSTMLayer(Config config) {
        this.config = config;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void build(List<Shape> inputShapes) {
        Shape inputShape = inputShapes.getFirst();

        List<Tensor> gates = new ArrayList<>();

        for (int i = 0; i < 4; i++) {
            gates.add(Tensors.orthogonal(config.hiddenDimension, config.hiddenDimension));
        }

        Tensor hiddenWeights = Tensors.concat(gates, 1).withGrad();
        Tensor weights = Tensors.zeros(inputShape.last(), 4 * config.hiddenDimension);
        Tensor bias = Tensors.zeros(4 * config.hiddenDimension);

        registerParam("hidden_weights", hiddenWeights);
        registerParam("weights", weights);
        registerParam("bias", bias);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void initWeights(List<Shape> inputShapes, RandomGenerator rng) {
        int inDim = inputShapes.getFirst().last();
        Tensor bias = param("bias");

        generateWeights("hidden_weights", rng, inDim, config.hiddenDimension);
        generateWeights("weights", rng, inDim, config.hiddenDimension);

        for (int i = 0; i < config.hiddenDimension; i++) {
            bias.set(1, i);
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public List<Shape> inferOutputShapes(List<Shape> inputShapes) {
        if (inputShapes.size() != 1) {
            throw Commons.illegalArgument("Layer requires 1 input but %s were given!", inputShapes.size());
        }

        Shape inputShape = inputShapes.getFirst();

        if (inputShape.rank() != 2) {
            throw Commons.illegalArgument("LSTM requires tensors with rank 2 but %s were given!", inputShape.rank());
        }

        if (!config.returnSequences) {
            return List.of(Shape.of(config.hiddenDimension));
        }

        return List.of(Shape.of(inputShape.last(1), config.hiddenDimension));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor input = inputs[0];

        if (input.rank() > 3) {
            throw Commons.illegalArgument("Expected input with rank <= 3 but got %s", input.rank());
        }

        while (input.rank() < 3) {
            input = input.unsqueeze();
        }

        int batch = input.shapeAt(0);
        int timesteps = input.shapeAt(1);

        Tensor weights = param("weights");
        Tensor hiddenWeights = param("hidden_weights");
        Tensor bias = param("bias");

        Tensor projection = input.matmul(weights);

        Tensor hiddenState = Tensors.zeros(batch, config.hiddenDimension).withGrad();
        Tensor cellState = Tensors.zeros(batch, config.hiddenDimension).withGrad();

        List<Tensor> hiddenStates = new ArrayList<>();

        Activation tanh = Activations.TANH.function();
        Activation sigmoid = Activations.SIGMOID.function();

        for (int t = 0; t < timesteps; t++) {
            Tensor timestep = projection.slice(Range.all(), Range.point(t), Range.all()).squeeze(1);
            Tensor hiddenProj = hiddenState.matmul(hiddenWeights);

            Tensor preActivation = timestep.add(hiddenProj).add(bias);

            Tensor forgetChunk = preActivation.slice(Range.all(), Range.interval(0, config.hiddenDimension));
            Tensor inputChunk = preActivation.slice(Range.all(), Range.interval(config.hiddenDimension, 2 * config.hiddenDimension));
            Tensor candidateChunk = preActivation.slice(Range.all(), Range.interval(2 * config.hiddenDimension, 3 * config.hiddenDimension));
            Tensor outputChunk = preActivation.slice(Range.all(), Range.interval(3 * config.hiddenDimension, 4 * config.hiddenDimension));

            Tensor f = forgetChunk.activate(sigmoid);
            Tensor i = inputChunk.activate(sigmoid);
            Tensor g = candidateChunk.activate(tanh);
            Tensor out = outputChunk.activate(sigmoid);

            cellState = f.mul(cellState).add(i.mul(g));
            hiddenState = out.mul(cellState.activate(tanh));

            if (config.returnSequences) {
                hiddenStates.add(hiddenState.reshape(batch, 1, config.hiddenDimension));
            }
        }

        Tensor result = hiddenState;

        if (config.returnSequences) {
            result = Tensors.concat(hiddenStates, 1);
        }

        return new Tensor[] { result };
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Layer copy() {
        LSTMLayer copy = new LSTMLayer(config);
        copyParameters(copy);

        return copy;
    }

    /**
     * Return the layer config.
     *
     * @return the config
     */
    public Config config() {
        return config;
    }
}
