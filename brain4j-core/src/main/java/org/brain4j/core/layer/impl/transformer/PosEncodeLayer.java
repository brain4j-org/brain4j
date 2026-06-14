package org.brain4j.core.layer.impl.transformer;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.Tensors;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.commons.Range;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.GpuTensor;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.random.RandomGenerator;

public class PosEncodeLayer extends Layer {

    public record Config(int length, int dimension) {}

    protected Config config;
    private final Map<Integer, Tensor> preGenerated = new HashMap<>();

    public PosEncodeLayer(int length, int dimension) {
        this(new Config(length, dimension));
    }

    public PosEncodeLayer(Config config) {
        this.config = config;

        for (int i = 0; i < config.length; i++) {
            preGenerated.put(i, generate(i, config.dimension));
        }
    }

    @Override
    public void build(List<Shape> inputShapes) {
    }

    @Override
    public void initWeights(List<Shape> inputShapes, RandomGenerator rng) {
    }

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
            Tensor add = preGenerated.computeIfAbsent(i, index -> generate(index, dimension));
            float[] addData = add.data();
            int index = i * dimension;
            System.arraycopy(addData, 0, posData, index, addData.length);
        }

        Tensor output = input.add(positional);

        if (input instanceof GpuTensor gpuTensor) output = output.to(gpuTensor.getDevice());
        if (input.usesGrad()) output = output.withGrad();

        return new Tensor[] { output };
    }

    @Override
    public Layer copy() {
        PosEncodeLayer copy = new PosEncodeLayer(config);
        // TODO
        return copy;
    }

    public void setWeights(Tensor weights) {
        this.config = new Config(weights.shapeAt(0), config.dimension);

        for (int i = 0; i < config.length; i++) {
            Tensor slice = weights.slice(Range.point(i), Range.all());
            preGenerated.put(i, slice.squeeze());
        }
    }

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

    public Config config() {
        return config;
    }
}
