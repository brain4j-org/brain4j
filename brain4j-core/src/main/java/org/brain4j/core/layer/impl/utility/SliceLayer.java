package org.brain4j.core.layer.impl.utility;

import java.util.List;
import java.util.random.RandomGenerator;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.commons.Range;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

/**
 * Slices inputs along fixed ranges.
 * Use it to extract crops or time windows from tensors.
 */
public class SliceLayer extends Layer {

    /**
     * Holds the slice ranges per dimension.
     */
    public record Config(Range[] ranges) {}

    protected final Config config;

    /**
     * Create a layer with the given ranges.
     */
    public SliceLayer(Range... ranges) {
        this(new Config(ranges));
    }

    /**
     * Create a layer with the given config.
     */
    public SliceLayer(Config config) {
        this.config = config;
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
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public List<Shape> inferOutputShapes(List<Shape> inputShapes) {
        if (inputShapes.isEmpty()) {
            throw Commons.illegalArgument("Layer requires at least 1 input but 0 were given!");
        }

        return inputShapes.stream().map(this::sliceShape).toList();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor[] result = new Tensor[inputs.length];

        for (int i = 0; i < inputs.length; i++) {
            Tensor input = inputs[i];
            int rank = input.rank();

            if (rank == config.ranges.length + 1) {
                Range[] expanded = new Range[config.ranges.length + 1];
                expanded[0] = Range.all();
                System.arraycopy(config.ranges, 0, expanded, 1, config.ranges.length);
                result[i] = input.slice(expanded);
            } else if (rank == config.ranges.length) {
                result[i] = input.slice(config.ranges);
            } else {
                throw Commons.illegalArgument("Slice requires rank %s or %s but %s was given!",
                    config.ranges.length, config.ranges.length + 1, rank);
            }
        }

        return result;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Layer copy() {
        return new SliceLayer(config);
    }

    /**
     * Return the slice ranges.
     *
     * @return the ranges
     */
    public Range[] ranges() {
        return config.ranges;
    }

    /**
     * Return the layer config.
     *
     * @return the config
     */
    public Config config() {
        return config;
    }

    private Shape sliceShape(Shape inputShape) {
        if (config.ranges.length != inputShape.rank()) {
            throw Commons.illegalArgument("Slice requires %s ranges but %s were given!",
                inputShape.rank(), config.ranges.length);
        }

        int[] dims = inputShape.dims();
        int[] out = new int[dims.length];

        for (int i = 0; i < dims.length; i++) {
            out[i] = config.ranges[i].size(dims[i]);
        }

        return Shape.of(out);
    }
}
