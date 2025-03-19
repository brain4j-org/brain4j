package net.echo.brain4j.utils.math.tensor;

import java.util.Arrays;
import java.util.Iterator;
import java.util.Random;
import java.util.function.Function;
import java.util.function.Supplier;

import net.echo.brain4j.utils.math.autograd.AutogradContext;
import net.echo.brain4j.utils.math.autograd.Operation;
import net.echo.brain4j.utils.math.autograd.operations.*;
import net.echo.brain4j.utils.math.tensor.index.Range;
import net.echo.brain4j.utils.math.vector.Vector;

public class Tensor implements Cloneable, Iterable<Double> {
    
    private final Vector data;
    private final int[] shape;
    private final int[] strides;
    private AutogradContext autogradContext;
    
    public Tensor(int... shape) {
        if (shape.length == 0) {
            throw new IllegalArgumentException("Shape cannot be empty");
        }
        
        this.shape = Arrays.copyOf(shape, shape.length);
        this.strides = computeStrides(shape);
        
        int size = computeSize(shape);
        this.data = new Vector(size);
    }
    
    private Tensor(Vector data, int[] shape, int[] strides) {
        this.data = data;
        this.shape = shape;
        this.strides = strides;
    }
    
    private static int computeSize(int[] shape) {
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        return size;
    }
    
    private static int[] computeStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }
    
    private int getLinearIndex(int... indices) {
        if (indices.length != shape.length) {
            throw new IllegalArgumentException(
                "The shape of the tensor does not match the number of indices"
            );
        }
        
        int linearIndex = 0;
        for (int i = 0; i < indices.length; i++) {
            if (indices[i] < 0 || indices[i] >= shape[i]) {
                throw new IndexOutOfBoundsException(
                    "Index " + indices[i] + " for dimension " + i + 
                    " is out of bounds [0, " + shape[i] + ")"
                );
            }
            linearIndex += indices[i] * strides[i];
        }
        return linearIndex;
    }
    
    public static Tensor of(int[] shape, float... data) {
        int size = computeSize(shape);
        if (data.length != size) {
            throw new IllegalArgumentException(
                "The length of the data (" + data.length + 
                ") does not match the shape dimension (" + size + ")"
            );
        }
        
        Tensor tensor = new Tensor(shape);
        for (int i = 0; i < data.length; i++) {
            tensor.data.set(i, data[i]);
        }
        return tensor;
    }
    
    public static Tensor of(int[] shape, double... data) {
        int size = computeSize(shape);
        if (data.length != size) {
            throw new IllegalArgumentException(
                "The length of the data (" + data.length + 
                ") does not match the shape dimension (" + size + ")"
            );
        }
        
        Tensor tensor = new Tensor(shape);
        for (int i = 0; i < data.length; i++) {
            tensor.data.set(i, (float) data[i]);
        }
        return tensor;
    }
    
    public static Tensor of(int[] shape, int... data) {
        int size = computeSize(shape);
        if (data.length != size) {
            throw new IllegalArgumentException(
                "The length of the data (" + data.length + 
                ") does not match the shape dimension (" + size + ")"
            );
        }
        
        Tensor tensor = new Tensor(shape);
        for (int i = 0; i < data.length; i++) {
            tensor.data.set(i, data[i]);
        }
        return tensor;
    }
    
    public static Tensor vector(float... data) {
        return of(new int[]{data.length}, data);
    }
    
    public static Tensor vector(Vector data) {
        int size = data.size();
        float[] floatData = new float[size];
        
        for (int i = 0; i < size; i++) {
            floatData[i] = data.get(i);
        }
        
        return of(new int[]{size}, floatData);
    }
    
    public static Tensor matrix(int rows, int cols, float... data) {
        return of(new int[]{rows, cols}, data);
    }
    
    public static Tensor zeros(int... shape) {
        Tensor tensor = new Tensor(shape);
        for (int i = 0; i < tensor.data.size(); i++) {
            tensor.data.set(i, 0.0f);
        }
        return tensor;
    }
    
    public static Tensor ones(int... shape) {
        Tensor tensor = new Tensor(shape);
        for (int i = 0; i < tensor.data.size(); i++) {
            tensor.data.set(i, 1.0f);
        }
        return tensor;
    }
    
    public static Tensor random(int... shape) {
        return random(new Random(), shape);
    }
    
    public static Tensor random(long seed, int... shape) {
        return random(new Random(seed), shape);
    }
    
    private static Tensor random(Random random, int... shape) {
        Tensor tensor = new Tensor(shape);
        for (int i = 0; i < tensor.data.size(); i++) {
            tensor.data.set(i, random.nextFloat());
        }
        return tensor;
    }
    
    public static Tensor uniform(double lowerBound, double upperBound, int... shape) {
        return uniform(new Random(), lowerBound, upperBound, shape);
    }
    
    public static Tensor uniform(long seed, double lowerBound, double upperBound, int... shape) {
        return uniform(new Random(seed), lowerBound, upperBound, shape);
    }
    
    private static Tensor uniform(Random random, double lowerBound, double upperBound, int... shape) {
        Tensor tensor = new Tensor(shape);
        double range = upperBound - lowerBound;
        for (int i = 0; i < tensor.data.size(); i++) {
            tensor.data.set(i, (float) (random.nextDouble() * range + lowerBound));
        }
        return tensor;
    }
    
    public static Tensor randn(double mean, double stddev, int... shape) {
        return randn(new Random(), mean, stddev, shape);
    }
    
    public static Tensor randn(long seed, double mean, double stddev, int... shape) {
        return randn(new Random(seed), mean, stddev, shape);
    }
    
    private static Tensor randn(Random random, double mean, double stddev, int... shape) {
        Tensor tensor = new Tensor(shape);
        for (int i = 0; i < tensor.data.size(); i++) {
            tensor.data.set(i, (float) (random.nextGaussian() * stddev + mean));
        }
        return tensor;
    }
    
    public int[] shape() {
        return Arrays.copyOf(shape, shape.length);
    }
    
    public int ndim() {
        return shape.length;
    }
    
    public int numel() {
        return data.size();
    }
    
    public Tensor set(double value, int... indices) {
        data.set(getLinearIndex(indices), (float) value);
        return this;
    }
    
    public float get(int... indices) {
        return data.get(getLinearIndex(indices));
    }
    
    public Tensor add(double value, int... indices) {
        data.set(getLinearIndex(indices), (float) value);
        return this;
    }
    
    public Tensor add(Tensor other) {
        checkSameShape(other);
        for (int i = 0; i < data.size(); i++) {
            data.set(i, data.get(i) + other.data.get(i));
        }
        return this;
    }
    
    public Tensor add(double value) {
        for (int i = 0; i < data.size(); i++) {
            data.set(i, data.get(i) + (float) value);
        }
        return this;
    }
    
    public Tensor plus(Tensor other) {
        return clone().add(other);
    }
    
    public Tensor plus(double value) {
        return clone().add(value);
    }
    
    public Tensor sub(Tensor other) {
        checkSameShape(other);
        for (int i = 0; i < data.size(); i++) {
            data.set(i, data.get(i) - other.data.get(i));
        }
        return this;
    }
    
    public Tensor sub(double value) {
        for (int i = 0; i < data.size(); i++) {
            data.set(i, data.get(i) - (float) value);
        }
        return this;
    }
    
    public Tensor minus(Tensor other) {
        return clone().sub(other);
    }
    
    public Tensor minus(double value) {
        return clone().sub(value);
    }
    
    public Tensor mul(Tensor other) {
        if (Arrays.equals(shape, other.shape)) {
            for (int i = 0; i < data.size(); i++) {
                data.set(i, data.get(i) * other.data.get(i));
            }
            return this;
        } else {
            return broadcastOperation(other, (a, b) -> a * b);
        }
    }
    
    public Tensor mul(double value) {
        for (int i = 0; i < data.size(); i++) {
            data.set(i, data.get(i) * (float) value);
        }
        return this;
    }
    
    public Tensor times(Tensor other) {
        return clone().mul(other);
    }
    
    public Tensor times(double value) {
        return clone().mul(value);
    }
    
    public Tensor div(Tensor other) {
        checkSameShape(other);
        for (int i = 0; i < data.size(); i++) {
            if (other.data.get(i) == 0) {
                throw new ArithmeticException("Division by zero");
            }
            data.set(i, data.get(i) / other.data.get(i));
        }
        return this;
    }
    
    public Tensor div(double value) {
        if (value == 0) {
            throw new ArithmeticException("Division by zero");
        }
        for (int i = 0; i < data.size(); i++) {
            data.set(i, data.get(i) / (float) value);
        }
        return this;
    }
    
    public Tensor divide(Tensor other) {
        return clone().div(other);
    }
    
    public Tensor divide(double value) {
        return clone().div(value);
    }
    
    public double sum() {
        double sum = 0;
        for (float value : data.toArray()) {
            sum += value;
        }
        return sum;
    }
    
    public double mean() {
        return sum() / data.size();
    }
    
    public double max() {
        double max = Double.NEGATIVE_INFINITY;
        for (float value : data.toArray()) {
            max = Math.max(max, value);
        }
        return max;
    }
    
    public double min() {
        double min = Double.POSITIVE_INFINITY;
        for (float value : data.toArray()) {
            min = Math.min(min, value);
        }
        return min;
    }
    
    public double dot(Tensor other) {
        checkSameShape(other);
        double sum = 0;
        for (int i = 0; i < data.size(); i++) {
            sum += data.get(i) * other.data.get(i);
        }
        return sum;
    }
    
    public double norm() {
        return Math.sqrt(normSquared());
    }
    
    public double normSquared() {
        double sum = 0;
        for (float value : data.toArray()) {
            sum += value * value;
        }
        return sum;
    }
    
    public Tensor normalize() {
        double norm = norm();
        if (norm > 0) {
            for (int i = 0; i < data.size(); i++) {
                data.set(i, data.get(i) / norm);
            }
        }
        return this;
    }
    
    public double distance(Tensor other) {
        return Math.sqrt(distanceSquared(other));
    }
    
    public double distanceSquared(Tensor other) {
        checkSameShape(other);
        double sum = 0;
        for (int i = 0; i < data.size(); i++) {
            double diff = data.get(i) - other.data.get(i);
            sum += diff * diff;
        }
        return sum;
    }
    
    public Tensor map(Function<Double, Double> function) {
        for (int i = 0; i < data.size(); i++) {
            data.set(i, function.apply((double) data.get(i)).floatValue());
        }
        return this;
    }
    
    public Tensor fill(double value) {
        for (int i = 0; i < data.size(); i++) {
            data.set(i, (float) value);
        }
        return this;
    }
    
    public Tensor fill(Supplier<Double> supplier) {
        for (int i = 0; i < data.size(); i++) {
            data.set(i, supplier.get().floatValue());
        }
        return this;
    }
    
    public float[] toArray() {
        return data.toArray();
    }
    
    public double[] toDoubleArray() {
        double[] result = new double[data.size()];
        for (int i = 0; i < data.size(); i++) {
            result[i] = data.get(i);
        }
        return result;
    }
    
    public Tensor reshape(int... newShape) {
        int newSize = computeSize(newShape);
        if (newSize != data.size()) {
            throw new IllegalArgumentException(
                "The total new dimension (" + newSize + 
                ") does not match the current dimension (" + data.size() + ")"
            );
        }
        
        return of(newShape, data.toArray());
    }
    
    public Tensor transpose() {
        if (shape.length != 2) {
            throw new UnsupportedOperationException(
                "transpose() is supported only for 2D tensors, not for tensors with " + shape.length + " dimensions"
            );
        }
        
        int rows = shape[0];
        int cols = shape[1];
        Tensor result = new Tensor(cols, rows);
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.set(get(i, j), j, i);
            }
        }
        
        return result;
    }
    
    public Tensor permute(int... dims) {
        if (dims.length != shape.length) {
            throw new IllegalArgumentException(
                "The number of dimensions in the permutation (" + dims.length +
                ") does not match the number of dimensions of the tensor (" + shape.length + ")"
            );
        }
        
        boolean[] dimUsed = new boolean[shape.length];
        for (int dim : dims) {
            if (dim < 0 || dim >= shape.length) {
                throw new IllegalArgumentException("Dimension out of bounds: " + dim);
            }
            if (dimUsed[dim]) {
                throw new IllegalArgumentException("Dimension duplicate in permutation: " + dim);
            }
            dimUsed[dim] = true;
        }
        
        int[] newShape = new int[shape.length];
        for (int i = 0; i < dims.length; i++) {
            newShape[i] = shape[dims[i]];
        }
        
        Tensor result = new Tensor(newShape);
        
        int[] indices = new int[shape.length];
        int[] newIndices = new int[shape.length];
        
        copyPermutedData(result, dims, indices, newIndices, 0);
        
        return result;
    }

    private void copyPermutedData(Tensor result, int[] dims, int[] indices, int[] newIndices, int dim) {
        if (dim == shape.length) {
            result.set(get(indices), newIndices);
            return;
        }
        
        for (int i = 0; i < shape[dim]; i++) {
            indices[dim] = i;
            newIndices[dims[dim]] = i;
            copyPermutedData(result, dims, indices, newIndices, dim + 1);
        }
    }
    
    private void checkSameShape(Tensor other) {
        if (!Arrays.equals(shape, other.shape)) {
            throw new IllegalArgumentException(
                "The shapes of the tensors do not match: " +
                Arrays.toString(shape) + " vs " + Arrays.toString(other.shape)
            );
        }
    }
    
    public Tensor select(int dim, int index) {
        if (dim < 0 || dim >= shape.length) {
            throw new IllegalArgumentException("Dimension out of bounds: " + dim);
        }
        if (index < 0 || index >= shape[dim]) {
            throw new IllegalArgumentException("Index out of bounds for dimension " + dim + ": " + index);
        }
        
        int[] newShape = new int[shape.length - 1];
        int newIdx = 0;
        for (int i = 0; i < shape.length; i++) {
            if (i != dim) {
                newShape[newIdx++] = shape[i];
            }
        }
        
        Tensor result = new Tensor(newShape);
        
        int[] indices = new int[shape.length];
        indices[dim] = index;
        int[] newIndices = new int[newShape.length];
        
        copySelectedData(result, dim, indices, newIndices, 0, 0);
        
        return result;
    }
    
    private void copySelectedData(Tensor result, int dim, int[] indices, int[] newIndices, int oldDim, int newDim) {
        if (oldDim == shape.length) {
            result.set(get(indices), newIndices);
            return;
        }
        
        if (oldDim == dim) {
            copySelectedData(result, dim, indices, newIndices, oldDim + 1, newDim);
        } else {
            for (int i = 0; i < shape[oldDim]; i++) {
                indices[oldDim] = i;
                newIndices[newDim] = i;
                copySelectedData(result, dim, indices, newIndices, oldDim + 1, newDim + 1);
            }
        }
    }
    
    public Tensor matmul(Tensor other) {
        if (shape.length != 2 || other.shape.length != 2) {
            throw new IllegalArgumentException("matmul requires 2D tensors");
        }
        
        int m = shape[0];
        int n = shape[1];
        int p = other.shape[1];
        
        if (n != other.shape[0]) {
            throw new IllegalArgumentException(
                "The inner dimensions do not match: " + n + " != " + other.shape[0]
            );
        }
        
        Tensor result = new Tensor(m, p);
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                float sum = 0;
                for (int k = 0; k < n; k++) {
                    sum += get(i, k) * other.get(k, j);
                }
                result.set(sum, i, j);
            }
        }
        
        return result;
    }

    @Override
    public String toString() {
        if (shape.length == 0) {
            return String.valueOf(data.get(0));
        }
        
        StringBuilder sb = new StringBuilder();
        appendTensor(sb, 0, new int[shape.length]);
        return sb.toString();
    }
    
    private void appendTensor(StringBuilder sb, int dim, int[] indices) {
        if (dim == shape.length - 1) {
            sb.append("[");
            for (int i = 0; i < shape[dim]; i++) {
                indices[dim] = i;
                sb.append(get(indices));
                if (i < shape[dim] - 1) {
                    sb.append(", ");
                }
            }
            sb.append("]");
        } else {
            sb.append("[");
            for (int i = 0; i < shape[dim]; i++) {
                indices[dim] = i;
                appendTensor(sb, dim + 1, indices);
                if (i < shape[dim] - 1) {
                    sb.append(",\n");
                    for (int j = 0; j <= dim; j++) {
                        sb.append(" ");
                    }
                }
            }
            sb.append("]");
        }
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        
        Tensor other = (Tensor) obj;
        if (!Arrays.equals(shape, other.shape)) return false;
        
        double epsilon = 1e-5;
        for (int i = 0; i < data.size(); i++) {
            if (Math.abs(data.get(i) - other.data.get(i)) > epsilon) {
                return false;
            }
        }
        
        return true;
    }
    
    @Override
    public int hashCode() {
        int result = Arrays.hashCode(shape);
        result = 31 * result + data.hashCode();
        return result;
    }
    
    @Override
    public Tensor clone() {
        return of(shape, data.toArray());
    }
    
    @Override
    public Iterator<Double> iterator() {
        return new Iterator<Double>() {
            private int currentIndex = 0;
            
            @Override
            public boolean hasNext() {
                return currentIndex < data.size();
            }
            
            @Override
            public Double next() {
                return (double) data.get(currentIndex++);
            }
        };
    }

    public static Tensor of(int[] shape, Vector data) {
        int size = computeSize(shape);
        if (data.size() != size) {
            throw new IllegalArgumentException(
                "The data length (" + data.size() + 
                ") does not match the shape dimension (" + size + ")"
            );
        }
        
        Tensor tensor = new Tensor(shape);
        for (int i = 0; i < data.size(); i++) {
            tensor.data.set(i, data.get(i));
        }
        return tensor;
    }

    public static Tensor matrix(int rows, int cols, Vector data) {
        return of(new int[]{rows, cols}, data);
    }

    public Tensor mul(Vector vec) {
        return mul(vector(vec));
    }

    public Tensor matmul(Vector vec) {
        if (shape.length != 2) {
            throw new IllegalArgumentException("matmul(Vector) requires a 2D tensor");
        }
        
        int m = shape[0];
        int n = shape[1];
        
        if (n != vec.size()) {
            throw new IllegalArgumentException(
                "The inner dimensions do not match: " + n + " != " + vec.size()
            );
        }
        
        Tensor result = new Tensor(m, 1);
        
        for (int i = 0; i < m; i++) {
            float sum = 0;
            for (int k = 0; k < n; k++) {
                sum += get(i, k) * vec.get(k);
            }
            result.set(sum, i, 0);
        }
        
        return result;
    }

    private Tensor broadcastOperation(Tensor other, java.util.function.BiFunction<Float, Float, Float> operation) {
        int[] resultShape = broadcastShapes(shape, other.shape);
        Tensor result = new Tensor(resultShape);
        
        int[] indices = new int[resultShape.length];
        broadcastFill(result, this, other, operation, indices, 0);
        
        return result;
    }

    private int[] broadcastShapes(int[] shape1, int[] shape2) {
        int maxDim = Math.max(shape1.length, shape2.length);
        int[] resultShape = new int[maxDim];
        
        for (int i = 0; i < maxDim; i++) {
            int dim1 = (i < shape1.length) ? shape1[shape1.length - 1 - i] : 1;
            int dim2 = (i < shape2.length) ? shape2[shape2.length - 1 - i] : 1;
            
            if (dim1 == 1 || dim2 == 1) {
                resultShape[maxDim - 1 - i] = Math.max(dim1, dim2);
            } else if (dim1 == dim2) {
                resultShape[maxDim - 1 - i] = dim1;
            } else {
                throw new IllegalArgumentException(
                    "Shapes cannot be broadcast: " + 
                    Arrays.toString(shape1) + " vs " + Arrays.toString(shape2)
                );
            }
        }
        
        return resultShape;
    }

    private void broadcastFill(Tensor result, Tensor a, Tensor b, 
                              java.util.function.BiFunction<Float, Float, Float> operation, 
                              int[] indices, int dim) {
        if (dim == result.shape.length) {
            int[] indicesA = mapIndicesToOperand(indices, a.shape);
            int[] indicesB = mapIndicesToOperand(indices, b.shape);
            
            float valueA = (indicesA != null) ? a.get(indicesA) : 0;
            float valueB = (indicesB != null) ? b.get(indicesB) : 0;
            result.set(operation.apply(valueA, valueB), indices);
            return;
        }
        
        for (int i = 0; i < result.shape[dim]; i++) {
            indices[dim] = i;
            broadcastFill(result, a, b, operation, indices, dim + 1);
        }
    }

    private int[] mapIndicesToOperand(int[] indices, int[] shape) {
        if (indices.length < shape.length) {
            return null;
        }
        
        int[] result = new int[shape.length];
        int offset = indices.length - shape.length;
        
        for (int i = 0; i < shape.length; i++) {
            int idx = indices[offset + i];
            if (idx >= shape[i]) {
                if (shape[i] == 1) {
                    result[i] = 0;
                } else {
                    return null;
                }
            } else {
                result[i] = idx;
            }
        }
        
        return result;
    }

    public Tensor sum(int dim, boolean keepDim) {
        if (dim < 0 || dim >= shape.length) {
            throw new IllegalArgumentException("Dimension " + dim + " out of bounds for tensor of shape " + Arrays.toString(shape));
        }
        
        int[] newShape = keepDim ? Arrays.copyOf(shape, shape.length) : new int[shape.length - 1];
        if (keepDim) {
            newShape[dim] = 1;
        } else {
            int newIdx = 0;
            for (int i = 0; i < shape.length; i++) {
                if (i != dim) {
                    newShape[newIdx++] = shape[i];
                }
            }
        }
        
        Tensor result = new Tensor(newShape);
        int[] indices = new int[shape.length];
        int[] resultIndices = keepDim ? new int[shape.length] : new int[shape.length - 1];
        
        sumAlongDimension(result, dim, keepDim, indices, resultIndices, 0);
        
        return result;
    }

    private void sumAlongDimension(Tensor result, int dim, boolean keepDim, int[] indices, int[] resultIndices, int currDim) {
        if (currDim == shape.length) {
            float value = get(indices);
            
            if (keepDim) {
                System.arraycopy(indices, 0, resultIndices, 0, indices.length);
                resultIndices[dim] = 0;
            } else {
                int resultIdx = 0;
                for (int i = 0; i < indices.length; i++) {
                    if (i != dim) {
                        resultIndices[resultIdx++] = indices[i];
                    }
                }
            }
            
            result.set(result.get(resultIndices) + value, resultIndices);
            return;
        }
        
        if (currDim == dim) {
            for (int i = 0; i < shape[currDim]; i++) {
                indices[currDim] = i;
                sumAlongDimension(result, dim, keepDim, indices, resultIndices, currDim + 1);
            }
        } else {
            for (int i = 0; i < shape[currDim]; i++) {
                indices[currDim] = i;
                sumAlongDimension(result, dim, keepDim, indices, resultIndices, currDim + 1);
            }
        }
    }

    public Tensor mean(int dim, boolean keepDim) {
        Tensor sumResult = sum(dim, keepDim);
        return sumResult.div((float) shape[dim]);
    }

    public Tensor view(int... newShape) {
        int autoIdx = -1;
        int knownSize = 1;
        for (int i = 0; i < newShape.length; i++) {
            if (newShape[i] == -1) {
                if (autoIdx >= 0) {
                    throw new IllegalArgumentException("Only one dimension can be -1");
                }
                autoIdx = i;
            } else {
                knownSize *= newShape[i];
            }
        }
        
        if (autoIdx >= 0) {
            int totalSize = data.size();
            if (totalSize % knownSize != 0) {
                throw new IllegalArgumentException(
                    "Total size " + totalSize + 
                    " is not divisible by the product of known dimensions " + knownSize
                );
            }
            newShape[autoIdx] = totalSize / knownSize;
        }
        
        return reshape(newShape);
    }

    public Tensor squeeze() {
        int nonSingletonDims = 0;
        for (int dim : shape) {
            if (dim != 1) {
                nonSingletonDims++;
            }
        }
        
        if (nonSingletonDims == shape.length) {
            return clone();
        }
        
        int[] newShape = new int[nonSingletonDims];
        int newIdx = 0;
        for (int dim : shape) {
            if (dim != 1) {
                newShape[newIdx++] = dim;
            }
        }
        
        if (newShape.length == 0) {
            newShape = new int[]{1};
        }
        
        return reshape(newShape);
    }

    public Tensor squeeze(int dim) {
        if (dim < 0 || dim >= shape.length) {
            throw new IllegalArgumentException("Dimension " + dim + " out of bounds");
        }
        
        if (shape[dim] != 1) {
            return clone();
        }
        
        int[] newShape = new int[shape.length - 1];
        int newIdx = 0;
        for (int i = 0; i < shape.length; i++) {
            if (i != dim) {
                newShape[newIdx++] = shape[i];
            }
        }
        
        return reshape(newShape);
    } 

    public Tensor unsqueeze(int dim) {
        if (dim < 0 || dim > shape.length) {
            throw new IllegalArgumentException("Dimension " + dim + " out of bounds");
        }
        
        int[] newShape = new int[shape.length + 1];
        
        System.arraycopy(shape, 0, newShape, 0, dim);
        
        newShape[dim] = 1;
        
        System.arraycopy(shape, dim, newShape, dim + 1, shape.length - dim);
        
        return reshape(newShape);
    }

    public Tensor slice(Range... ranges) {
        if (ranges.length > shape.length) {
            throw new IllegalArgumentException("Too many ranges specified");
        }
        
        int[] newShape = new int[shape.length];
        for (int i = 0; i < shape.length; i++) {
            if (i < ranges.length && ranges[i] != null) {
                newShape[i] = ranges[i].size(shape[i]);
            } else {
                newShape[i] = shape[i];
            }
        }
        
        Tensor result = new Tensor(newShape);
        
        int[] srcIndices = new int[shape.length];
        int[] dstIndices = new int[shape.length];
        
        sliceCopy(result, ranges, srcIndices, dstIndices, 0);
        
        return result;
    }

    private void sliceCopy(Tensor result, Range[] ranges, int[] srcIndices, int[] dstIndices, int dim) {
        if (dim == shape.length) {
            result.set(get(srcIndices), dstIndices);
            return;
        }
        
        Range range = dim < ranges.length ? ranges[dim] : null;
        int start = 0;
        int end = shape[dim];
        int step = 1;
        
        if (range != null) {
            start = range.start(shape[dim]);
            end = range.end(shape[dim]);
            step = range.step();
        }
        
        for (int i = start, j = 0; i < end; i += step, j++) {
            srcIndices[dim] = i;
            dstIndices[dim] = j;
            sliceCopy(result, ranges, srcIndices, dstIndices, dim + 1);
        }
    }

    public Tensor requiresGrad(boolean requiresGrad) {
        if (autogradContext == null) {
            autogradContext = new AutogradContext(requiresGrad);
        } else {
            autogradContext = new AutogradContext(requiresGrad);
        }
        return this;
    }

    public boolean requiresGrad() {
        return autogradContext != null && autogradContext.requiresGrad();
    }

    public Tensor grad() {
        if (autogradContext != null) {
            return autogradContext.getGrad();
        }
        return null;
    }

    public void backward() {
        backward(ones(shape));
    }

    public void backward(Tensor gradOutput) {
        if (autogradContext != null) {
            autogradContext.backward(gradOutput);
        }
    }

    /**
     * Addition with autograd support
     * 
     * @param other Tensor to add
     * @return A new tensor resulting from the addition
     */
    public Tensor addWithGrad(Tensor other) {
        if (!requiresGrad() && !other.requiresGrad()) {
            return plus(other);
        }
        
        Operation op = new AddOperation();
        Tensor result = op.forward(this, other);
        
        if (result.autogradContext == null) {
            result.autogradContext = new AutogradContext(true);
        }
        result.autogradContext.setOperation(op, this, other);
        
        return result;
    }

    /**
     * Multiplication with autograd support
     * 
     * @param other Tensor to multiply
     * @return A new tensor resulting from the multiplication
     */
    public Tensor mulWithGrad(Tensor other) {
        if (!requiresGrad() && !other.requiresGrad()) {
            return times(other);
        }
        
        Operation op = new MulOperation();
        Tensor result = op.forward(this, other);
        
        if (result.autogradContext == null) {
            result.autogradContext = new AutogradContext(true);
        }
        result.autogradContext.setOperation(op, this, other);
        
        return result;
    }

    /**
     * Subtraction with autograd support
     * 
     * @param other Tensor to subtract
     * @return A new tensor resulting from the subtraction
     */
    public Tensor subWithGrad(Tensor other) {
        if (!requiresGrad() && !other.requiresGrad()) {
            return minus(other);
        }
        
        Operation op = new SubOperation();
        Tensor result = op.forward(this, other);
        
        if (result.autogradContext == null) {
            result.autogradContext = new AutogradContext(true);
        }
        result.autogradContext.setOperation(op, this, other);
        
        return result;
    }

    /**
     * Division with autograd support
     * 
     * @param other Tensor to divide by
     * @return A new tensor resulting from the division
     */
    public Tensor divWithGrad(Tensor other) {
        if (!requiresGrad() && !other.requiresGrad()) {
            return divide(other);
        }
        
        Operation op = new DivOperation();
        Tensor result = op.forward(this, other);
        
        if (result.autogradContext == null) {
            result.autogradContext = new AutogradContext(true);
        }
        result.autogradContext.setOperation(op, this, other);
        
        return result;
    }

    /**
     * Matrix multiplication with autograd support
     * 
     * @param other Tensor to multiply
     * @return A new tensor resulting from the matrix multiplication
     */
    public Tensor matmulWithGrad(Tensor other) {
        if (!requiresGrad() && !other.requiresGrad()) {
            return matmul(other);
        }
        
        Operation op = new MatMulOperation();
        Tensor result = op.forward(this, other);
        
        if (result.autogradContext == null) {
            result.autogradContext = new AutogradContext(true);
        }
        result.autogradContext.setOperation(op, this, other);
        
        return result;
    }
} 