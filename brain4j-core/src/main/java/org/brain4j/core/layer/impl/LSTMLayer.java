package org.brain4j.core.layer.impl;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.commons.Range;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

public class LSTMLayer extends Layer {

    public record Config(int hiddenDimension, boolean returnSequences) {}

    protected final Config config;

    public LSTMLayer(int hiddenDimension, boolean returnSequences) {
        this(new Config(hiddenDimension, returnSequences));
    }

    public LSTMLayer(Config config) {
        this.config = config;
    }

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

    @Override
    public void initWeights(List<Shape> inputShapes, RandomGenerator rng) {
        int inDim = inputShapes.getFirst().last();
        Tensor bias = getParam("bias");

        generateWeights("hidden_weights", rng, inDim, config.hiddenDimension);
        generateWeights("weights", rng, inDim, config.hiddenDimension);

        for (int i = 0; i < config.hiddenDimension; i++)
            bias.set(1, i);
    }

    @Override
    public List<Shape> inferOutputShapes(List<Shape> inputShapes) {
        if (inputShapes.size() != 1) {
            throw Commons.illegalArgument("Layer requires 1 input but %s were given!", inputShapes.size());
        }

        // [..., timesteps, dim]
        Shape inputShape = inputShapes.getFirst();

        if (inputShape.rank() != 2) {
            throw Commons.illegalArgument("LSTM requires tensors with rank 2 but %s were given!", inputShape.rank());
        }
        if (!config.returnSequences) {
            return List.of(Shape.of(config.hiddenDimension));
        }

        return List.of(Shape.of(inputShape.last(1), config.hiddenDimension));
    }

    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        // [batch, timesteps, dimension]
        Tensor input = inputs[0];

        if (input.rank() > 3) {
            throw Commons.illegalArgument("Expected input with rank <= 3 but got %s", input.rank());
        }

        while (input.rank() < 3) {
            input = input.unsqueeze();
        }

        int batch = input.shapeAt(0);
        int timesteps = input.shapeAt(1);

        Tensor weights = getParam("weights");
        Tensor hiddenWeights = getParam("hidden_weights");
        Tensor bias = getParam("bias");

        // [batch, timesteps, 4 * hidden_dim]
        Tensor projection = input.matmulGrad(weights);

        // [batch, timesteps, hidden_size]
        Tensor hiddenState = Tensors.zeros(batch, config.hiddenDimension).withGrad();
        Tensor cellState = Tensors.zeros(batch, config.hiddenDimension).withGrad();

        List<Tensor> hiddenStates = new ArrayList<>();

        Activation tanh = Activations.TANH.function();
        Activation sigmoid = Activations.SIGMOID.function();

        for (int t = 0; t < timesteps; t++) {
            Tensor timestep = projection.sliceGrad(Range.all(), Range.point(t), Range.all()).squeezeGrad(1);
            Tensor hiddenProj = hiddenState.matmulGrad(hiddenWeights); // [batch, 4 * hidden_dim]

            Tensor preActivation = timestep.addGrad(hiddenProj).addGrad(bias); // [batch, 4 * hidden_dim]

            Tensor forgetChunk = preActivation.sliceGrad(Range.all(), Range.interval(0, config.hiddenDimension));
            Tensor inputChunk = preActivation.sliceGrad(Range.all(), Range.interval(config.hiddenDimension, 2 * config.hiddenDimension));
            Tensor candidateChunk = preActivation.sliceGrad(Range.all(), Range.interval(2 * config.hiddenDimension, 3 * config.hiddenDimension));
            Tensor outputChunk = preActivation.sliceGrad(Range.all(), Range.interval(3 * config.hiddenDimension, 4 * config.hiddenDimension));

            Tensor f = forgetChunk.activateGrad(sigmoid);
            Tensor i = inputChunk.activateGrad(sigmoid);
            Tensor g = candidateChunk.activateGrad(tanh);
            Tensor out = outputChunk.activateGrad(sigmoid);

            cellState = f.mulGrad(cellState).addGrad(i.mulGrad(g));
            hiddenState = out.mulGrad(cellState.activateGrad(tanh));

            if (config.returnSequences) {
                hiddenStates.add(hiddenState.reshapeGrad(batch, 1, config.hiddenDimension));
            }
        }

        // [batch, timesteps, hidden_dim]
        Tensor result = hiddenState;

        if (config.returnSequences) {
            result = Tensors.concatGrad(hiddenStates, 1);
        }

        return new Tensor[] { result };
    }

    @Override
    public Layer copy() {
        LSTMLayer copy = new LSTMLayer(config);
        copyParameters(copy);
        return copy;
    }

    public Config config() {
        return config;
    }
}
