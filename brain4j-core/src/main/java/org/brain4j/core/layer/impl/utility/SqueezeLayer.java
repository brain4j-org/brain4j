package org.brain4j.core.layer.impl.utility;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

public class SqueezeLayer extends Layer {
    
    private final int dimension;
    
    public SqueezeLayer(int dimension) {
        this.dimension = dimension;
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
        
        return inputShapes.stream().map(this::squeezeShape).toList();
    }

    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor[] results = new Tensor[inputs.length];
        
        for (int i = 0; i < inputs.length; i++) {
            Tensor input = inputs[i];
            
            if (dimension == -1) {
                int[] shape = input.shape();
                int[] newShape = new int[shape.length];
                int count = 0;
                
                for (int d = 0; d < shape.length; d++) {
                    if (d == 0 || shape[d] != 1) {
                        newShape[count++] = shape[d];
                    }
                }
                
                if (count == shape.length) {
                    results[i] = input;
                } else {
                    int[] compact = new int[count];
                    System.arraycopy(newShape, 0, compact, 0, count);
                    results[i] = input.reshapeGrad(compact);
                }
            } else {
                int dim = dimension;
                
                if (dimension >= 0 && input.rank() > 1) {
                    dim = dimension + 1;
                }
                
                results[i] = input.squeezeGrad(dim);
            }
        }
        
        return results;
    }

    @Override
    public Layer copy() {
        return new SqueezeLayer(dimension);
    }
    
    public int dimension() {
        return dimension;
    }
    
    private Shape squeezeShape(Shape inputShape) {
        if (dimension == -1) {
            List<Integer> kept = new ArrayList<>();
            
            for (int dim : inputShape.dims()) {
                if (dim != 1) kept.add(dim);
            }
            
            int[] out = new int[kept.size()];
            
            for (int i = 0; i < kept.size(); i++) {
                out[i] = kept.get(i);
            }
            
            return Shape.of(out);
        }
        
        int rank = inputShape.rank();
        int dimIndex = dimension < 0 ? Math.floorMod(dimension, rank) : dimension;
        
        if (dimIndex < 0 || dimIndex >= rank) {
            throw Commons.illegalArgument("Squeeze dimension %s is out of range for rank %s", dimension, rank);
        }
        
        if (inputShape.dim(dimIndex) != 1) {
            throw Commons.illegalArgument("Dimension %s is not 1 and cannot be squeezed", dimIndex);
        }
        
        int[] dims = inputShape.dims();
        int[] out = new int[dims.length - 1];
        
        for (int i = 0, j = 0; i < dims.length; i++) {
            if (i == dimIndex) continue;
            out[j++] = dims[i];
        }
        
        return Shape.of(out);
    }
}
