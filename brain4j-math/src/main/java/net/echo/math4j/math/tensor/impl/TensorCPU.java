package net.echo.math4j.math.tensor.impl;

import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;
import net.echo.math4j.math.tensor.autograd.AutogradContext;
import net.echo.math4j.math.tensor.autograd.Operation;
import net.echo.math4j.math.tensor.autograd.operations.*;
import net.echo.math4j.math.tensor.index.Range;
import net.echo.math4j.math.tensor.ops.Convolution;
import net.echo.math4j.math.vector.Vector;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Random;
import java.util.SplittableRandom;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;

public class TensorCPU implements Cloneable, Tensor {
    
    private final Vector data;
    private final int[] shape;
    private final int[] strides;
    private AutogradContext autogradContext;
    
    public TensorCPU(int... shape) {
        if (shape.length == 0) {
            throw new IllegalArgumentException("Shape cannot be empty");
        }
        
        this.shape = Arrays.copyOf(shape, shape.length);
        this.strides = computeStrides(shape);
        
        int size = computeSize(shape);
        this.data = new Vector(size);
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
            throw new IllegalArgumentException("The shape of the tensor does not match the number of indices");
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
        Tensor tensor = new TensorCPU(shape);
        
        for (int i = 0; i < data.length; i++) {
            tensor.getData().set(i, data[i]);
        }
        
        return tensor;
    }
    
    public static Tensor of(int[] shape, double... data) {
        Tensor tensor = new TensorCPU(shape);
        
        for (int i = 0; i < data.length; i++) {
            tensor.getData().set(i, (float) data[i]);
        }
        
        return tensor;
    }
    
    public static Tensor of(int[] shape, int... data) {
        Tensor tensor = new TensorCPU(shape);

        for (int i = 0; i < data.length; i++) {
            tensor.getData().set(i, data[i]);
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
        Tensor tensor = new TensorCPU(shape);

        for (int i = 0; i < tensor.getData().size(); i++) {
            tensor.getData().set(i, 0.0f);
        }

        return tensor;
    }
    
    public static Tensor ones(int... shape) {
        Tensor tensor = new TensorCPU(shape);

        for (int i = 0; i < tensor.getData().size(); i++) {
            tensor.getData().set(i, 1.0f);
        }

        return tensor;
    }
    
    public static Tensor random(int... shape) {
        return random(Random.from(new SplittableRandom()), shape);
    }
    
    public static Tensor random(long seed, int... shape) {
        return random(Random.from(new SplittableRandom(seed)), shape);
    }
    
    private static Tensor random(Random random, int... shape) {
        Tensor tensor = new TensorCPU(shape);

        for (int i = 0; i < tensor.getData().size(); i++) {
            tensor.getData().set(i, random.nextFloat());
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
        Tensor tensor = new TensorCPU(shape);
        double range = upperBound - lowerBound;

        for (int i = 0; i < tensor.getData().size(); i++) {
            tensor.getData().set(i, (float) (random.nextDouble() * range + lowerBound));
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
        Tensor tensor = new TensorCPU(shape);

        for (int i = 0; i < tensor.getData().size(); i++) {
            tensor.getData().set(i, (float) (random.nextGaussian() * stddev + mean));
        }

        return tensor;
    }

    @Override
    public AutogradContext getAutogradContext() {
        return autogradContext;
    }

    @Override
    public void setAutogradContext(AutogradContext autogradContext) {
        this.autogradContext = autogradContext;
    }

    @Override
    public Vector getData() {
        return data;
    }

    public int[] shape() {
        return Arrays.copyOf(shape, shape.length);
    }
    
    public int dimension() {
        return shape.length;
    }
    
    public int elements() {
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
            data.set(i, data.get(i) + other.getData().get(i));
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
            data.set(i, data.get(i) - other.getData().get(i));
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
        if (Arrays.equals(shape, other.shape())) {
            for (int i = 0; i < data.size(); i++) {
                data.set(i, data.get(i) * other.getData().get(i));
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
            if (other.getData().get(i) == 0) {
                throw new ArithmeticException("Division by zero");
            }

            data.set(i, data.get(i) / other.getData().get(i));
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
            sum += data.get(i) * other.getData().get(i);
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
            double diff = data.get(i) - other.getData().get(i);
            sum += diff * diff;
        }

        return sum;
    }

    @Override
    public Tensor mapWithIndex(BiFunction<Integer, Double, Double> function) {
        for (int i = 0; i < data.size(); i++) {
            double value = data.get(i);
            data.set(i, function.apply(i, value));
        }

        return this;
    }

    public Tensor map(Function<Double, Double> function) {
        for (int i = 0; i < data.size(); i++) {
            double value = data.get(i);
            data.set(i, function.apply(value));
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
        if (dimension() == 1) {
            return reshape(1, elements());
        }

        if (shape.length != 2) {
            throw new UnsupportedOperationException("transpose() is supported only for 2D tensors, not for tensors with "
                    + shape.length + " dimensions");
        }
        
        int rows = shape[0];
        int cols = shape[1];

        Tensor result = TensorFactory.matrix(cols, rows);
        
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
        
        Tensor result = new TensorCPU(newShape);
        
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
        if (!Arrays.equals(shape, other.shape())) {
            throw new IllegalArgumentException(
                "The shapes of the tensors do not match: " +
                Arrays.toString(shape) + " vs " + Arrays.toString(other.shape())
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
        
        Tensor result = new TensorCPU(newShape);
        
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

    public Tensor matmulFast(Tensor other) {
        if (shape.length != 2 || other.shape().length != 2) {
            throw new IllegalArgumentException("matmul richiede tensori 2D");
        }

        final int m = shape[0];
        final int n = shape[1];
        final int p = other.shape()[1];

        if (n != other.shape()[0]) {
            throw new IllegalArgumentException("Le dimensioni interne non corrispondono: " + n + " != " + other.shape()[0]);
        }

        float[] A = this.getData().toArray();
        float[] B = other.getData().toArray();
        Tensor result = new TensorCPU(m, p);
        float[] C = result.getData().toArray();

        final int blockSize = 64;

        for (int iBlock = 0; iBlock < m; iBlock += blockSize) {
            int iMax = Math.min(iBlock + blockSize, m);

            for (int kBlock = 0; kBlock < n; kBlock += blockSize) {
                int kMax = Math.min(kBlock + blockSize, n);

                for (int jBlock = 0; jBlock < p; jBlock += blockSize) {
                    int jMax = Math.min(jBlock + blockSize, p);

                    for (int i = iBlock; i < iMax; i++) {
                        int indexA_i = i * n;
                        int indexC_i = i * p + jBlock;

                        for (int k = kBlock; k < kMax; k++) {
                            float aVal = A[indexA_i + k];

                            int indexB_k = k * p + jBlock;
                            int indexC = indexC_i;

                            for (int j = jBlock; j < jMax; j++) {
                                C[indexC++] += aVal * B[indexB_k++];
                            }
                        }
                    }
                }
            }
        }

        return result;
    }
    
    public Tensor matmul(Tensor other) {
        if (shape.length != 2 || other.shape().length != 2) {
            throw new IllegalArgumentException("matmul requires 2D tensors");
        }

        int m = shape[0];
        int n = shape[1];
        int p = other.shape()[1];

        if (n != other.shape()[0]) {
            throw new IllegalArgumentException("The inner dimensions do not match: " + n + " != " + other.shape()[0]);
        }

        Tensor result = new TensorCPU(m, p);

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

    private void appendTensor(StringBuilder sb, int dim, int[] indices, String format) {
        if (dim == shape.length - 1) {
            sb.append("[");
            for (int i = 0; i < shape[dim]; i++) {
                indices[dim] = i;

                float value = get(indices);
                sb.append(String.format(format, value));

                if (i < shape[dim] - 1) {
                    sb.append(", ");
                }
            }
            sb.append("]");
        } else {
            sb.append("[");
            for (int i = 0; i < shape[dim]; i++) {
                indices[dim] = i;
                appendTensor(sb, dim + 1, indices, format);

                if (i < shape[dim] - 1) {
                    sb.append(",\n");
                    sb.append(" ".repeat(dim + 1));
                }
            }

            sb.append("]");
        }
    }

    public static Tensor of(int[] shape, Vector data) {
        int size = computeSize(shape);
        if (data.size() != size) {
            throw new IllegalArgumentException(
                "The data length (" + data.size() + 
                ") does not match the shape dimension (" + size + ")"
            );
        }
        
        Tensor tensor = new TensorCPU(shape);
        for (int i = 0; i < data.size(); i++) {
            tensor.getData().set(i, data.get(i));
        }
        return tensor;
    }

    public static Tensor matrix(int rows, int cols, Vector data) {
        return of(new int[]{rows, cols}, data);
    }

    public Tensor pow(double value) {
        return map(x -> Math.pow(x, value));
    }

    public Tensor pow(Tensor other) {
        return mapWithIndex((i, x) -> Math.pow(x, other.get(i)));
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
        
        Tensor result = new TensorCPU(m, 1);
        
        for (int i = 0; i < m; i++) {
            float sum = 0;

            for (int k = 0; k < n; k++) {
                sum += get(i, k) * vec.get(k);
            }

            result.set(sum, i, 0);
        }
        
        return result;
    }

    private Tensor broadcastOperation(Tensor other, BiFunction<Float, Float, Float> operation) {
        int[] resultShape = broadcastShapes(shape, other.shape());
        Tensor result = new TensorCPU(resultShape);
        
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

    private void broadcastFill(Tensor result, Tensor a, Tensor b, BiFunction<Float, Float, Float> operation,
                               int[] indices, int dim) {
        if (dim == result.shape().length) {
            int[] indicesA = mapIndicesToOperand(indices, a.shape());
            int[] indicesB = mapIndicesToOperand(indices, b.shape());
            
            float valueA = (indicesA != null) ? a.get(indicesA) : 0;
            float valueB = (indicesB != null) ? b.get(indicesB) : 0;
            result.set(operation.apply(valueA, valueB), indices);
            return;
        }
        
        for (int i = 0; i < result.shape()[dim]; i++) {
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
        
        Tensor result = new TensorCPU(newShape);
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

    @Override
    public Tensor setChannel(int channel, Tensor other) {
        if (dimension() != 3) {
            throw new IllegalArgumentException("Tensor must be 3-dimensional!");
        }

        if (channel <= 0 || channel >= shape[0]) {
            throw new IllegalArgumentException("Invalid channel index: " + channel);
        }

        int height = shape[1];
        int width = shape[2];

        float[] sliceData = other.getData().toArray();
        int offset = channel * height * width;

        System.arraycopy(sliceData, 0, data.toArray(), offset, height * width);

        return null;
    }

    @Override
    public Tensor slice(int channel) {
        if (dimension() != 3) {
            throw new IllegalArgumentException("Tensor must be 3-dimensional!");
        }

        if (channel < 0 || channel >= shape[0]) {
            throw new IllegalArgumentException("Invalid channel index: " + channel);
        }

        int height = shape[1];
        int width = shape[2];

        float[] sliceData = new float[height * width];
        int offset = channel * height * width;

        System.arraycopy(data.toArray(), offset, sliceData, 0, height * width);

        return TensorFactory.of(new int[]{height, width}, sliceData);
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
        
        Tensor result = new TensorCPU(newShape);
        
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

    public Tensor addWithGrad(Tensor other) {
        if (!requiresGrad() && !other.requiresGrad()) {
            return plus(other);
        }

        Operation op = new AddOperation();
        Tensor result = op.forward(this, other);

        if (result.getAutogradContext() == null) {
            result.setAutogradContext(new AutogradContext(true));
        }

        result.getAutogradContext().setOperation(op, this, other);

        return result;
    }

    public Tensor mulWithGrad(Tensor other) {
        if (!requiresGrad() && !other.requiresGrad()) {
            return times(other);
        }
        
        Operation op = new MulOperation();
        Tensor result = op.forward(this, other);
        
        if (result.getAutogradContext() == null) {
            result.setAutogradContext(new AutogradContext(true));
        }

        result.getAutogradContext().setOperation(op, this, other);
        
        return result;
    }

    public Tensor subWithGrad(Tensor other) {
        if (!requiresGrad() && !other.requiresGrad()) {
            return minus(other);
        }
        
        Operation op = new SubOperation();
        Tensor result = op.forward(this, other);
        
        if (result.getAutogradContext() == null) {
            result.setAutogradContext(new AutogradContext(true));
        }

        result.getAutogradContext().setOperation(op, this, other);
        
        return result;
    }

    public Tensor divWithGrad(Tensor other) {
        if (!requiresGrad() && !other.requiresGrad()) {
            return divide(other);
        }
        
        Operation op = new DivOperation();
        Tensor result = op.forward(this, other);
        
        if (result.getAutogradContext() == null) {
            result.setAutogradContext(new AutogradContext(true));
        }

        result.getAutogradContext().setOperation(op, this, other);
        
        return result;
    }

    public Tensor matmulWithGrad(Tensor other) {
        if (!requiresGrad() && !other.requiresGrad()) {
            return matmul(other);
        }
        
        Operation op = new MatMulOperation();
        Tensor result = op.forward(this, other);
        
        if (result.getAutogradContext() == null) {
            result.setAutogradContext(new AutogradContext(true));
        }

        result.getAutogradContext().setOperation(op, this, other);
        
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;

        Tensor other = (Tensor) obj;
        if (!Arrays.equals(shape, other.shape())) return false;

        double epsilon = 1e-5;
        for (int i = 0; i < data.size(); i++) {
            if (Math.abs(data.get(i) - other.getData().get(i)) > epsilon) {
                return false;
            }
        }

        return true;
    }

    @Override
    public Tensor clone() {
        return of(shape, data.toArray());
    }

    @Override
    public Iterator<Double> iterator() {
        return new Iterator<>() {
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


    @Override
    public String toString() {
        if (shape.length == 0) {
            return String.valueOf(data.get(0));
        }

        StringBuilder sb = new StringBuilder();
        appendTensor(sb, 0, new int[shape.length], "%f");

        return sb.toString();
    }

    @Override
    public String toString(String format) {
        if (shape.length == 0) {
            return String.format(format, data.get(0));
        }

        StringBuilder sb = new StringBuilder();
        appendTensor(sb, 0, new int[shape.length], format);

        return sb.toString();
    }

    @Override
    public void serialize(DataOutputStream stream) throws IOException {
        stream.writeInt(dimension());
        
        for (int i = 0; i < shape.length; i++) {
            stream.writeInt(shape[i]);
        }

        stream.writeInt(data.size());

        for (int i = 0; i < data.size(); i++) {
            stream.writeDouble(data.get(i));
        }
    }

    @Override
    public Tensor deserialize(DataInputStream stream) throws IOException {
        int[] shape = new int[stream.readInt()];

        for (int i = 0; i < shape.length; i++) {
            shape[i] = stream.readInt();
        }

        Tensor data = of(shape);
        int dataSize = stream.readInt();

        for (int i = 0; i < dataSize; i++) {
            data.getData().set(i, stream.readDouble());
        }

        return data;
    }

    @Override
    public Tensor softmax() {
        return softmax(1);
    }

    @Override
    public Tensor softmax(double temperature) {
        int dim = dimension() > 1 ? 1 : 0;

        if (dim >= dimension()) {
            throw new IllegalArgumentException("Dimension " + dim + " out of bounds for limits of tensor shape " + Arrays.toString(shape));
        }

        Tensor result = clone();

        if (dimension() == 1) {
            softmax1D(temperature, result);
        } else if (dimension() == 2) {
            if (dim == 0) {
                softmaxColumns(temperature, result);
            } else {
                softmaxRows(temperature, result);
            }
        } else {
            throw new UnsupportedOperationException("Softmax operation is only supported for 1D/2D tensors.");
        }

        return result;
    }

    @Override
    public Tensor gpu() {
        return TensorGPU.fromTensor(this);
    }

    @Override
    public Tensor cpu() {
        return this;
    }

    @Override
    public boolean checkNaN() {
        for (int i = 0; i < data.size(); i++) {
            if (Float.isNaN(data.get(i))) {
                return true;
            }
        }

        return false;
    }

    @Override
    public Tensor sqrt() {
        return pow(0.5);
    }

    private void softmax1D(double temperature, Tensor tensor) {
        double max = Double.NEGATIVE_INFINITY;

        for (int i = 0; i < tensor.elements(); i++) {
            max = Math.max(max, tensor.get(i));
        }

        double sum = 0.0;

        for (int i = 0; i < tensor.elements(); i++) {
            sum += Math.exp((tensor.get(i) - max) / temperature);
        }

        for (int i = 0; i < tensor.elements(); i++) {
            double value = Math.exp((tensor.get(i) - max) / temperature) / sum;
            tensor.set(value, i);
        }
    }

    private void softmaxRows(double temperature, Tensor tensor) {
        int rows = tensor.shape()[0];
        int cols = tensor.shape()[1];

        for (int i = 0; i < rows; i++) {
            double max = Double.NEGATIVE_INFINITY;

            for (int j = 0; j < cols; j++) {
                max = Math.max(max, tensor.get(i, j));
            }

            double sum = 0.0;

            for (int j = 0; j < cols; j++) {
                sum += Math.exp((tensor.get(i, j) - max) / temperature);
            }

            for (int j = 0; j < cols; j++) {
                double value = Math.exp((tensor.get(i, j) - max) / temperature) / sum;
                tensor.set(value, i, j);
            }
        }
    }
    
    private void softmaxColumns(double temperature, Tensor tensor) {
        int rows = tensor.shape()[0];
        int cols = tensor.shape()[1];

        for (int j = 0; j < cols; j++) {
            double max = Double.NEGATIVE_INFINITY;

            for (int i = 0; i < rows; i++) {
                max = Math.max(max, tensor.get(i, j));
            }

            double sum = 0.0;

            for (int i = 0; i < rows; i++) {
                sum += Math.exp((tensor.get(i, j) - max) / temperature);
            }

            for (int i = 0; i < rows; i++) {
                double value = Math.exp((tensor.get(i, j) - max) / temperature) / sum;
                tensor.set(value, i, j);
            }
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor convolve(Tensor kernel) {
        int dim = this.dimension();

        if (dim > 3) {
            throw new IllegalArgumentException("Convolution is supported only for 1D and 2D tensors");
        }

        if (kernel.dimension() != dim) {
            throw new IllegalArgumentException("The kernel dimension must match the input dimension");
        }
        
        Convolution.ConvolutionType convType;
        if (this.elements() > 1000 || kernel.elements() > 100) { // convType based on tensor size
            convType = Convolution.ConvolutionType.FFT;
        } else {
            convType = Convolution.ConvolutionType.DIRECT;
        }

        if (dim == 1) {
            return Convolution.convolve1D(this, kernel, Convolution.PaddingMode.SAME, convType);
        } else if (dim == 2) {
            return Convolution.convolve2D(this, kernel, Convolution.PaddingMode.SAME, convType);
        } else {
            throw new UnsupportedOperationException("Not implemented yet.");
        }
    }
}