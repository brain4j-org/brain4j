package org.brain4j.math.tensor;

import org.brain4j.math.Copyable;
import org.brain4j.math.Tensors;
import org.brain4j.math.commons.Commons;

import java.util.Arrays;

public class Shape implements Copyable<Shape> {

    private final int[] dims;

    protected Shape(int[] dims) {
        for (int dim : dims) {
            if (dim < 0) {
                throw Commons.illegalArgument("Dimension at %s is negative!", dim);
            }
        }

        this.dims = dims.clone();
    }

    public static Shape of(int... dims) {
        return new Shape(dims);
    }
    
    public static Shape concat(Shape... all) {
        Shape base = all[0];
        
        for (int i = 1; i < all.length; i++) {
            base = base.concat(all[i], -1);
        }
        
        return base;
    }
    
    @Override
    public Shape copy() {
        return new Shape(dims);
    }
    
    public Shape concat(Shape other, int dim) {
        if (rank() != other.rank()) {
            throw Commons.illegalArgument("Shapes must have the same rank");
        }
        
        int rank = rank();
        int dimension = Math.floorMod(dim, rank);
        
        if (dimension < 0 || dimension >= rank) {
            throw Commons.illegalArgument("Invalid dimension: " + dimension);
        }
        
        for (int i = 0; i < rank; i++) {
            if (i != dimension && dims[i] != other.dims()[i]) {
                throw Commons.illegalArgument("Shapes must match in all dims except the concatenation one. " +
                    "Current: %s, Other: %s", Arrays.toString(dims), Arrays.toString(other.dims()));
            }
        }
        
        int[] newShape = Arrays.copyOf(dims, rank);
        newShape[dimension] += other.dims()[dimension];
        
        return new Shape(newShape);
    }
    
    public int dim(int index) {
        if (index < 0) {
            index = Math.floorMod(index, dims.length);
        }
        
        return dims[index];
    }

    public int last() {
        return dims[dims.length - 1];
    }

    public int last(int offset) {
        if (offset > dims.length - 1) {
            throw new IllegalArgumentException("Offset cannot be higher than rank - 1");
        }

        return dims[dims.length - 1 - offset];
    }
    
    public int size() {
        return Tensors.computeSize(dims);
    }

    public int rank() {
        return dims.length;
    }

    public int[] dims() {
        return dims;
    }

    public Shape slice(int start, int end) {
        end = Math.floorMod(end, dims.length); // support for negative indices
        
        if (end < start) {
            throw new IllegalArgumentException("End must be greater or equal than start!");
        }
        
        int[] result = new int[end - start];
        System.arraycopy(dims, start, result, 0, result.length);
        return Shape.of(result);
    }
    
    public void copy(int[] result, int destOffset) {
        if (rank() > result.length) throw Commons.illegalArgument("Rank is higher than result array");
        System.arraycopy(dims, 0, result, destOffset, dims.length);
    }
    
    public void copy(int[] result) {
        copy(result, 0);
    }
}
