package org.brain4j.core.layer.impl.transformer;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.random.RandomGenerator;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.Tensors;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.commons.Range;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.GpuTensor;

/**
 * Adds sinusoidal positional encodings to embeddings.
 * Use it after embeddings to inject sequence order.
 */
public class PosEncodeLayer extends Layer {

    /**
     * Holds max length and embedding width.
     */
    public record Config(int length, int dimension) {}

    private final Map<Integer, Tensor> embeddings = new HashMap<>();
    protected final Config config;

    /**
     * Create a layer with the given config.
     */
    public PosEncodeLayer(Config config) {
        this.config = config;
    }

    /**
     * Create a layer with the given length and width.
     */
    public PosEncodeLayer(int length, int dimension) {
        this(new Config(length, dimension));
    }

    /**
     * Create a layer initialized from the given weight table.
     *
     * @param weights the table with shape [length, dimension]
     */
    public PosEncodeLayer(int length, int dimension, Tensor weights) {
        this.config = new Config(length, dimension);

        for (int i = 0; i < config.length; i++) {
            Tensor slice = weights.slice(Range.point(i), Range.all());
            embeddings.put(i, slice.squeeze());
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void build(List<Shape> inputShapes) {
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void initWeights(List<Shape> inputShapes, RandomGenerator rng) {
        for (int i = 0; i < config.length; i++) {
            embeddings.put(i, generate(i, config.dimension));
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
            throw Commons.illegalArgument("Positional encoding requires tensors with rank 2 but %s were given!",
                inputShape.rank());
        }

        if (inputShape.last() != config.dimension) {
            throw Commons.illegalArgument("Expected embedding dim %s but got %s", config.dimension, inputShape.last());
        }

        return List.of(inputShape);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor input = inputs[0];
        int[] shape = input.shape();

        if (shape.length != 3) {
            throw Commons.illegalArgument("Input must have shape [batch, seq_length, dimension]! Got: %s",
                Arrays.toString(shape));
        }

        int seqLength = shape[1];
        int dimension = shape[2];

        Tensor positional = Tensors.zeros(seqLength, dimension);
        float[] posData = positional.data();

        for (int i = 0; i < seqLength; i++) {
            Tensor add = embeddings.computeIfAbsent(i, index -> generate(index, dimension));
            float[] addData = add.data();
            int index = i * dimension;
            System.arraycopy(addData, 0, posData, index, addData.length);
        }

        Tensor output = input.add(positional);

        if (input instanceof GpuTensor gpuTensor) {
            output = output.to(gpuTensor.device());
        }

        return new Tensor[] { output };
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Layer copy() {
        PosEncodeLayer copy = new PosEncodeLayer(config);
        copy.embeddings.putAll(embeddings);

        return copy;
    }

    /**
     * Generate the sinusoidal encoding for one position.
     *
     * @param embeddingDim the encoding width, must be positive
     * @return the encoding row with shape [1, embeddingDim]
     */
    public Tensor generate(int position, int embeddingDim) {
        Tensor token = Tensors.zeros(embeddingDim);

        for (int i = 0; i < embeddingDim; i++) {
            double exponent = (2.0 * Math.floor(i / 2.0)) / embeddingDim;

            double angle = position / Math.pow(10000, exponent);
            double value = (i % 2 == 0) ? Math.sin(angle) : Math.cos(angle);

            token.set(value, i);
        }

        return token.reshape(1, embeddingDim);
    }

    /**
     * Return the layer config.
     *
     * @return the config
     */
    public Config config() {
        return config;
    }

    /**
     * Return cached positional rows by index.
     *
     * @return the embeddings map
     */
    public Map<Integer, Tensor> embeddings() {
        return embeddings;
    }
}
