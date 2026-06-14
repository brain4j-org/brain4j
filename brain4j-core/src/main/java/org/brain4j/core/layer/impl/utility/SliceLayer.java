package org.brain4j.core.layer.impl.utility;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.commons.Range;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

public class SliceLayer extends Layer {

    public record Config(Range[] ranges) {}

    protected Config config;

    public SliceLayer(Range... ranges) {
        this(new Config(ranges));
    }

    public SliceLayer(Config config) {
        this.config = config;
    }

    @Override
    public void build(List<Shape> inputShapes) {
    }

    @Override
    public void initWeights(List<Shape> inputShapes, RandomGenerator rng) {
    }

    @Override
    public List<Shape> inferOutputShapes(List<Shape> inputShapes) {
        if (inputShapes.isEmpty()) {
            throw Commons.illegalArgument("Layer requires at least 1 input but 0 were given!");
        }

        return inputShapes.stream().map(this::sliceShape).toList();
    }

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
                result[i] = input.sliceGrad(expanded);
            } else if (rank == config.ranges.length) {
                result[i] = input.sliceGrad(config.ranges);
            } else {
                throw Commons.illegalArgument("Slice requires rank %s or %s but %s was given!",
                    config.ranges.length, config.ranges.length + 1, rank);
            }
        }

        return result;
    }

    @Override
    public Layer copy() {
        return new SliceLayer(config);
    }

    public Range[] ranges() {
        return config.ranges;
    }

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
