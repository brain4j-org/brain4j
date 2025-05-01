package net.echo.math.tensor.impl;

import net.echo.math.activation.Activation;
import net.echo.math.tensor.Tensor;
import net.echo.math.tensor.Tensors;
import net.echo.math.tensor.autograd.AutogradContext;
import net.echo.math.tensor.autograd.Operation;
import net.echo.math.tensor.autograd.operations.*;
import net.echo.math.tensor.impl.cpu.ParallelMatmul;
import net.echo.math.tensor.index.Range;
import net.echo.math.tensor.ops.Convolution;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Random;
import java.util.SplittableRandom;
import java.util.concurrent.ForkJoinPool;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;

public class TensorCPU implements Cloneable, Tensor {

    private static final ForkJoinPool POOL = ForkJoinPool.commonPool();

    protected float[] data;
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
        this.data = new float[size];
    }

    private int computeSize(int[] shape) {
        int size = 1;
        
        for (int dim : shape) {
            size *= dim;
        }
        
        return size;
    }
    
    private int[] computeStrides(int[] shape) {
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
            tensor.getData()[i] = data[i];
        }
        
        return tensor;
    }
    
    public static Tensor of(int[] shape, double... data) {
        Tensor tensor = new TensorCPU(shape);
        
        for (int i = 0; i < data.length; i++) {
            tensor.getData()[i] = (float) data[i];
        }
        
        return tensor;
    }
    
    public static Tensor of(int[] shape, int... data) {
        Tensor tensor = new TensorCPU(shape);
        
        for (int i = 0; i < data.length; i++) {
            tensor.getData()[i] = (float) data[i];
        }

        return tensor;
    }
    
    public static Tensor ones(int... shape) {
        Tensor tensor = new TensorCPU(shape);
        Arrays.fill(tensor.getData(), 1.0f);
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

        for (int i = 0; i < tensor.getData().length; i++) {
            tensor.getData()[i] = random.nextFloat();
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

        for (int i = 0; i < tensor.getData().length; i++) {
            tensor.getData()[i] = (float) (random.nextDouble() * range + lowerBound);
        }

        return tensor;
    }
    
    public static Tensor randn(double mean, double stddev, int... shape) {
        return randn(new Random(), mean, stddev, shape);
    }
    
    public static Tensor randn(long seed, double mean, double stddev, int... shape) {
        return randn(new Random(seed), mean, stddev, shape);
    }

    public static Tensor randn(Random random, double mean, double stddev, int... shape) {
        Tensor tensor = new TensorCPU(shape);

        for (int i = 0; i < tensor.getData().length; i++) {
            tensor.getData()[i] = (float) (random.nextGaussian() * stddev + mean);
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
    public float[] getData() {
        return data;
    }

    @Override
    public int[] shape() {
        return Arrays.copyOf(shape, shape.length);
    }

    @Override
    public int dimension() {
        return shape.length;
    }

    @Override
    public int elements() {
        return data.length;
    }

    @Override
    public Tensor set(double value, int... indices) {
        data[getLinearIndex(indices)] = (float) value;
        return this;
    }

    @Override
    public float get(int... indices) {
        return data[getLinearIndex(indices)];
    }

    @Override
    public Tensor add(double value, int... indices) {
        data[getLinearIndex(indices)] = (float) value;
        return this;
    }

    @Override
    public Tensor add(Tensor other) {
        checkSameShape(other);

        for (int i = 0; i < data.length; i++) {
            data[i] += other.getData()[i];
        }

        return this;
    }

    @Override
    public Tensor add(double value) {
        for (int i = 0; i < data.length; i++) {
            data[i] += (float) value;
        }

        return this;
    }

    @Override
    public Tensor plus(Tensor other) {
        return clone().add(other);
    }

    @Override
    public Tensor plus(double value) {
        return clone().add(value);
    }

    @Override
    public Tensor sub(Tensor other) {
        checkSameShape(other);

        for (int i = 0; i < data.length; i++) {
            data[i] = data[i] - other.getData()[i];
        }

        return this;
    }

    @Override
    public Tensor sub(double value) {
        for (int i = 0; i < data.length; i++) {
            data[i] = data[i] - (float) value;
        }

        return this;
    }

    @Override
    public Tensor minus(Tensor other) {
        return clone().sub(other);
    }

    @Override
    public Tensor minus(double value) {
        return clone().sub(value);
    }

    @Override
    public Tensor mul(Tensor other) {
        checkSameShape(other);

        for (int i = 0; i < data.length; i++) {
            data[i] = data[i] * other.getData()[i];
        }

        return this;
    }

    @Override
    public Tensor mul(double value) {
        for (int i = 0; i < data.length; i++) {
            data[i] = data[i] * (float) value;
        }

        return this;
    }

    @Override
    public Tensor times(Tensor other) {
        return clone().mul(other);
    }

    @Override
    public Tensor times(double value) {
        return clone().mul(value);
    }

    @Override
    public Tensor div(Tensor other) {
        checkSameShape(other);

        for (int i = 0; i < data.length; i++) {
            if (other.getData()[i] == 0) {
                throw new ArithmeticException("Division by zero");
            }

            data[i] = data[i] / other.getData()[i];
        }

        return this;
    }

    @Override
    public Tensor div(double value) {
        if (value == 0) {
            throw new ArithmeticException("Division by zero");
        }

        for (int i = 0; i < data.length; i++) {
            data[i] = data[i] / (float) value;
        }

        return this;
    }

    @Override
    public Tensor divide(Tensor other) {
        return clone().div(other);
    }

    @Override
    public Tensor divide(double value) {
        return clone().div(value);
    }

    @Override
    public double sum() {
        double sum = 0;

        for (float value : data) {
            sum += value;
        }

        return sum;
    }

    @Override
    public double mean() {
        return sum() / data.length;
    }

    @Override
    public double variance() {
        double mean = mean();
        double variance = 0.0;

        for (float value : data) {
            variance += Math.pow(value - mean, 2);
        }

        return variance / data.length;
    }

    @Override
    public double max() {
        double max = Double.NEGATIVE_INFINITY;

        for (float value : data) {
            max = Math.max(max, value);
        }

        return max;
    }

    @Override
    public double min() {
        double min = Double.POSITIVE_INFINITY;

        for (float value : data) {
            min = Math.min(min, value);
        }

        return min;
    }

    @Override
    public double dot(Tensor other) {
        checkSameShape(other);
        double sum = 0;

        for (int i = 0; i < data.length; i++) {
            sum += data[i] * other.getData()[i];
        }

        return sum;
    }

    @Override
    public double norm() {
        return Math.sqrt(normSquared());
    }

    @Override
    public double normSquared() {
        double sum = 0;

        for (float value : data) {
            sum += value * value;
        }

        return sum;
    }

    @Override
    public Tensor normalize() {
        double norm = norm();

        if (norm > 0) {
            for (int i = 0; i < data.length; i++) {
                data[i] /= (float) norm;
            }
        }

        return this;
    }

    @Override
    public double distance(Tensor other) {
        return Math.sqrt(distanceSquared(other));
    }

    @Override
    public double distanceSquared(Tensor other) {
        checkSameShape(other);
        double sum = 0;

        for (int i = 0; i < data.length; i++) {
            double diff = data[i] - other.getData()[i];
            sum += diff * diff;
        }

        return sum;
    }

    @Override
    public Tensor mapWithIndex(BiFunction<Integer, Float, Float> function) {
        for (int i = 0; i < data.length; i++) {
            float value = data[i];
            data[i] = function.apply(i, value);
        }

        return this;
    }

    @Override
    public Tensor map(Function<Double, Double> function) {
        for (int i = 0; i < data.length; i++) {
            double value = data[i];
            data[i] = function.apply(value).floatValue();
        }

        return this;
    }

    @Override
    public Tensor fill(float value) {
        Arrays.fill(data, value);
        return null;
    }

    @Override
    public Tensor fill(Supplier<Double> supplier) {
        for (int i = 0; i < data.length; i++) {
            data[i] = supplier.get().floatValue();
        }

        return this;
    }

    @Override
    public Tensor reshape(int... newShape) {
        int newSize = computeSize(newShape);

        if (newSize != data.length) {
            throw new IllegalArgumentException(
                "The total new dimension (" + newSize +
                ") does not match the current dimension (" + data.length + ")"
            );
        }

        return of(newShape, data);
    }

    @Override
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

        Tensor result = Tensors.matrix(cols, rows);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.set(get(i, j), j, i);
            }
        }

        return result;
    }

    @Override
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

    @Override
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

    @Override
    public Tensor matmul(Tensor other) {
        if (shape.length < 2 || other.shape().length < 2) {
            throw new IllegalArgumentException("Matrix multiplication requires at least 2D tensors!");
        }

        if (shape.length != other.shape().length) {
            throw new IllegalArgumentException(
                    "Dimensions do not match: " + shape.length + " != " + other.shape().length
            );
        }

        for (int i = 0; i < shape.length - 2; i++) {
            if (shape[i] != other.shape()[i]) {
                throw new IllegalArgumentException(
                        "Batch dimensions do not match at index " + i + ": " + shape[i] + " != " + other.shape()[i]
                );
            }
        }

        int dims = shape.length;

        int m = shape[dims - 2];
        int n = shape[dims - 1];

        int k = other.shape()[dims - 2];
        int p = other.shape()[dims - 1];

        if (n != k) {
            throw new IllegalArgumentException("Inner dimensions must match: " + n + " != " + k);
        }

        int[] resultShape = new int[dims];

        int batch = 1;
        for (int i = 0; i < dims - 2; i++) {
            resultShape[i] = shape[i];
            batch *= shape[i];
        }

        resultShape[dims - 2] = m;
        resultShape[dims - 1] = p;

        Tensor result = new TensorCPU(resultShape);

        float[] A = this.getData();
        float[] B = other.getData();
        float[] C = result.getData();

        ParallelMatmul.multiply(batch, m, n, p, A, B, C, POOL);
        return result;
    }

    private void appendTensor(StringBuilder sb, int dim, int[] indices, String format) {
        if (dim == shape.length - 1) {
            sb.append("[");
            for (int i = 0; i < shape[dim]; i++) {
                indices[dim] = i;

                float value = get(indices);
                sb.append(format.formatted(value));

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

    @Override
    public Tensor pow(double value) {
        return map(x -> Math.pow(x, value));
    }

    @Override
    public Tensor pow(Tensor other) {
        return mapWithIndex((i, x) -> (float) Math.pow(x, other.get(i)));
    }

    @Override
    public Tensor sqrt() {
        return pow(0.5);
    }

    @Override
    public Tensor vector() {
        return reshape(elements());
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

    @Override
    public Tensor mean(int dim, boolean keepDim) {
        Tensor sumResult = sum(dim, keepDim);
        return sumResult.div((float) shape[dim]);
    }

    @Override
    public Tensor sign() {
        Tensor result = of(shape);

        for (int i = 0; i < data.length; i++) {
            if (data[i] > 0) {
                result.getData()[i] = 1;
            } else if (data[i] < 0) {
                result.getData()[i] = -1;
            } else {
                result.getData()[i] = 0;
            }
        }

        return result;
    }

    @Override
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
            int totalSize = data.length;
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

    @Override
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

    @Override
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

    @Override
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

        float[] sliceData = other.getData();
        int offset = channel * height * width;

        System.arraycopy(sliceData, 0, data, offset, height * width);

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

        System.arraycopy(data, offset, sliceData, 0, height * width);

        return Tensors.of(new int[]{height, width}, sliceData);
    }

    @Override
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

    @Override
    public Tensor withGrad() {
        this.autogradContext = new AutogradContext(true);
        return this;
    }

    @Override
    public boolean usesGrad() {
        return autogradContext != null && autogradContext.requiresGrad();
    }

    @Override
    public void zerograd() {
        if (autogradContext != null) {
            autogradContext.zerograd();
        }
    }

    @Override
    public Tensor grad() {
        if (autogradContext != null) {
            return autogradContext.getGrad();
        }

        return null;
    }

    @Override
    public void backward() {
        backward(ones(shape));
    }

    @Override
    public void backward(Tensor gradOutput) {
        if (autogradContext == null) {
            throw new IllegalArgumentException("Autograd is not enabled for this tensor");
        }

        autogradContext.backward(gradOutput);
    }

    @Override
    public Tensor forward(Operation op, Tensor other) {
        Tensor result = op.forward(this, other);

        if (result.getAutogradContext() == null) {
            result.setAutogradContext(new AutogradContext(true));
        }

        result.getAutogradContext().setOperation(op, this, other);
        return result;
    }

    @Override
    public Tensor addWithGrad(Tensor other) {
        if (!usesGrad() && !other.usesGrad()) {
            return plus(other);
        }

        return forward(new AddOperation(), other);
    }

    @Override
    public Tensor mulWithGrad(Tensor other) {
        if (!usesGrad() && !other.usesGrad()) {
            return times(other);
        }

        return forward(new MulOperation(), other);
    }

    @Override
    public Tensor subWithGrad(Tensor other) {
        if (!usesGrad() && !other.usesGrad()) {
            return minus(other);
        }

        return forward(new SubOperation(), other);
    }

    @Override
    public Tensor divWithGrad(Tensor other) {
        if (!usesGrad() && !other.usesGrad()) {
            return divide(other);
        }

        return forward(new DivOperation(), other);
    }

    @Override
    public Tensor matmulWithGrad(Tensor other) {
        if (!usesGrad() && !other.usesGrad()) {
            return matmul(other);
        }

        return forward(new MatMulOperation(), other);
    }

    @Override
    public Tensor activateWithGrad(Activation activation) {
        if (!usesGrad()) {
            return activation.activate(this);
        }

        return forward(new ActivationOperation(activation), null);
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;

        Tensor other = (Tensor) obj;

        if (!Arrays.equals(shape, other.shape())) return false;

        double epsilon = 1e-5;

        for (int i = 0; i < data.length; i++) {
            if (Math.abs(data[i] - other.getData()[i]) > epsilon) {
                return false;
            }
        }

        return true;
    }

    @Override
    public int argmax() {
        double max = Double.NEGATIVE_INFINITY;
        int maxIndex = 0;

        for (int i = 0; i < data.length; i++) {
            if (data[i] > max) {
                max = data[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    @Override
    public Tensor clone() {
        return of(shape, data);
    }

    @Override
    public Iterator<Float> iterator() {
        return new Iterator<>() {
            private int currentIndex = 0;

            @Override
            public boolean hasNext() {
                return currentIndex < data.length;
            }

            @Override
            public Float next() {
                return data[currentIndex++];
            }
        };
    }

    @Override
    public String toString() {
        if (shape.length == 0) {
            return String.valueOf(data[0]);
        }

        StringBuilder sb = new StringBuilder();
        appendTensor(sb, 0, new int[shape.length], "%f");

        return sb.toString();
    }

    @Override
    public String toString(String format) {
        if (shape.length == 0) {
            return format.formatted(data[0]);
        }

        StringBuilder sb = new StringBuilder();
        appendTensor(sb, 0, new int[shape.length], format);

        return sb.toString();
    }

    @Override
    public void serialize(DataOutputStream stream) throws Exception {
        stream.writeInt(dimension());

        for (int i : shape) {
            stream.writeInt(i);
        }

        stream.writeInt(data.length);

        for (float element : data) {
            stream.writeFloat(element);
        }
    }

    @Override
    public Tensor deserialize(DataInputStream stream) throws Exception {
        int[] shape = new int[stream.readInt()];

        for (int i = 0; i < shape.length; i++) {
            shape[i] = stream.readInt();
        }

        Tensor data = of(shape);
        int dataSize = stream.readInt();

        for (int i = 0; i < dataSize; i++) {
            data.getData()[i] = stream.readFloat();
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
            softmaxRows(temperature, result);
        } else {
            throw new UnsupportedOperationException("Softmax operation is only supported for 1D/2D tensors.");
        }

        return result;
    }

    @Override
    public boolean checkNaN() {
        for (float element : data) {
            if (Float.isNaN(element)) {
                return true;
            }
        }

        return false;
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