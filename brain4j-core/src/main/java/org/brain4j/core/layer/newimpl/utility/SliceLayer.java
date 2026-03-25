package org.brain4j.core.layer.newimpl.utility;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.commons.Range;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

public class SliceLayer extends Layer {
    
    private final Range[] ranges;
    
    public SliceLayer(Range... ranges) {
        this.ranges = ranges;
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
            
            if (rank == ranges.length + 1) {
                Range[] expanded = new Range[ranges.length + 1];
                expanded[0] = Range.all();
                System.arraycopy(ranges, 0, expanded, 1, ranges.length);
                result[i] = input.sliceGrad(expanded);
            } else if (rank == ranges.length) {
                result[i] = input.sliceGrad(ranges);
            } else {
                throw Commons.illegalArgument("Slice requires rank %s or %s but %s was given!",
                    ranges.length, ranges.length + 1, rank);
            }
        }
        
        return result;
    }

    @Override
    public Layer copy() {
        return new SliceLayer(ranges);
    }
    
    private Shape sliceShape(Shape inputShape) {
        if (ranges.length != inputShape.rank()) {
            throw Commons.illegalArgument("Slice requires %s ranges but %s were given!",
                inputShape.rank(), ranges.length);
        }
        
        int[] dims = inputShape.dims();
        int[] out = new int[dims.length];
        
        for (int i = 0; i < dims.length; i++) {
            out[i] = ranges[i].size(dims[i]);
        }
        
        return Shape.of(out);
    }
}
