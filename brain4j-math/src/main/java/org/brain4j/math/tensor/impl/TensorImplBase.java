package org.brain4j.math.tensor.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.lang.DoubleToDoubleFunction;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;
import org.brain4j.math.tensor.autograd.AutogradContext;
import org.brain4j.math.tensor.autograd.Operation;
import org.brain4j.math.tensor.autograd.operations.*;
import org.brain4j.math.tensor.broadcast.TensorBroadcast;
import org.brain4j.math.tensor.index.Range;

import java.util.Arrays;
import java.util.function.BiFunction;
import java.util.function.Supplier;

public abstract class TensorImplBase implements Tensor, Cloneable {

    protected AutogradContext autogradContext;
    protected int[] shape;
    protected int[] strides;
    protected float[] data;

    protected int getLinearIndex(int... indices) {
        if (indices.length != shape.length) {
            throw new IllegalArgumentException("The shape of the tensor does not match the number of indices");
        }

        int linearIndex = 0;

        for (int i = 0; i < indices.length; i++) {
            if (indices[i] < 0 || indices[i] >= shape[i]) {
                throw new IndexOutOfBoundsException(
                        "Index " + indices[i] + " for dimension " + i + " is out of bounds [0, " + shape[i] + ")"
                );
            }

            linearIndex += indices[i] * strides[i];
        }

        return linearIndex;
    }

    protected int computeSize(int[] shape) {
        int size = 1;

        for (int dim : shape) {
            size *= dim;
        }

        return size;
    }

    protected void checkSameShape(Tensor other) {
        if (!Arrays.equals(shape, other.shape())) {
            throw new IllegalArgumentException(
                    "The shapes of the tensors do not match: " +
                            Arrays.toString(shape) + " vs " + Arrays.toString(other.shape())
            );
        }
    }

    protected void appendTensor(StringBuilder result, int dim, int[] indices, String format) {
        if (dim == shape.length - 1) {
            result.append("[");
            for (int i = 0; i < shape[dim]; i++) {
                indices[dim] = i;

                float value = get(indices);
                result.append(format.formatted(value));

                if (i < shape[dim] - 1) {
                    result.append(", ");
                }
            }
            result.append("]");
        } else {
            result.append("[");
            for (int i = 0; i < shape[dim]; i++) {
                indices[dim] = i;
                appendTensor(result, dim + 1, indices, format);

                if (i < shape[dim] - 1) {
                    result.append(",\n");
                    result.append(" ".repeat(dim + 1));
                }
            }

            result.append("]");
        }
    }

    @Override
    public int[] shape() {
        return shape;
    }

    @Override
    public float[] data() {
        return data;
    }

    @Override
    public int[] strides() {
        return strides;
    }

    @Override
    public float get(int... indices) {
        return data[getLinearIndex(indices)];
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
    public Tensor set(double value, int... indices) {
        data[getLinearIndex(indices)] = (float) value;
        return this;
    }

    @Override
    public Tensor add(double value, int... indices) {
        data[getLinearIndex(indices)] += (float) value;
        return this;
    }

    @Override
    public Tensor clone() {
        try {
            TensorImplBase copy = (TensorImplBase) super.clone();

            copy.shape = shape.clone();
            copy.strides = strides.clone();
            copy.data = data.clone();

            return copy;
        } catch (CloneNotSupportedException e) {
            throw new AssertionError("Clone not supported", e);
        }
    }

    @Override
    public Tensor add(Tensor other) {
        return TensorBroadcast.add(this, other);
    }

    @Override
    public Tensor add(double value) {
        for (int i = 0; i < data.length; i++) {
            data[i] += (float) value;
        }

        return this;
    }

    @Override
    public Tensor sub(Tensor other) {
        return TensorBroadcast.sub(this, other);
    }

    @Override
    public Tensor sub(double value) {
        for (int i = 0; i < data.length; i++) {
            data[i] -= (float) value;
        }

        return this;
    }

    @Override
    public Tensor mul(Tensor other) {
        return TensorBroadcast.mul(this, other);
    }

    @Override
    public Tensor mul(double value) {
        for (int i = 0; i < data.length; i++) {
            data[i] *= (float) value;
        }

        return this;
    }

    @Override
    public Tensor div(Tensor other) {
        return TensorBroadcast.div(this, other);
    }

    @Override
    public Tensor div(double value) {
        for (int i = 0; i < data.length; i++) {
            data[i] /= (float) value;
        }

        return this;
    }

    @Override
    public Tensor pow(double value) {
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) Math.pow(data[i], value);
        }

        return this;
    }

    @Override
    public Tensor pow(Tensor other) {
        return TensorBroadcast.pow(this, other);
    }

    @Override
    public Tensor sqrt() {
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) Math.sqrt(data[i]);
        }

        return this;
    }

    @Override
    public Tensor vector() {
        return reshape(elements());
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
            double diff = data[i] - other.data()[i];
            sum += diff * diff;
        }

        return sum;
    }

    @Override
    public Tensor transpose() {
        if (dimension() == 1) {
            return reshape(1, elements());
        }

        if (shape.length != 2) {
            throw new UnsupportedOperationException(
                    "transpose() is supported only for 2D tensors, not for tensors with " + shape.length + " dimensions"
            );
        }

        int rows = shape[0];
        int cols = shape[1];

        Tensor result = Tensors.matrix(cols, rows);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.set(get(i, j), j, i);
            }
        }

        if (usesGrad()) {
            result.setAutogradContext(autogradContext);
        }

        return result;
    }

    @Override
    public double sum() {
        double sum = 0.0;

        for (float value : data) {
            sum += value;
        }

        return sum;
    }

    @Override
    public double mean() {
        return sum() / elements();
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

    private int[] computeNewShape(int[] shape, int dim, boolean keepDim) {
        int[] newShape = keepDim ? Arrays.copyOf(shape, shape.length) : new int[shape.length - 1];

        if (keepDim) {
            newShape[dim] = 1;
        } else {
            for (int i = 0, j = 0; i < shape.length; i++) {
                if (i != dim) {
                    newShape[j++] = shape[i];
                }
            }
        }

        return newShape;
    }

    @Override
    public Tensor sum(int dim, boolean keepDim) {
        if (dim < 0 || dim >= shape.length) {
            throw new IllegalArgumentException("Dimension " + dim + " out of bounds for tensor of shape " + Arrays.toString(shape));
        }

        int[] newShape = computeNewShape(shape, dim, keepDim);
        int reducedSize = shape[dim];

        Tensor result = Tensors.create(newShape);
        float[] resultData = result.data();

        int outerSize = 1;

        for (int i = 0; i < dim; i++) {
            outerSize *= shape[i];
        }

        int innerSize = 1;

        for (int i = dim + 1; i < shape.length; i++) {
            innerSize *= shape[i];
        }

        for (int outer = 0; outer < outerSize; outer++) {
            for (int inner = 0; inner < innerSize; inner++) {
                float sum = 0;

                for (int i = 0; i < reducedSize; i++) {
                    int index = outer * reducedSize * innerSize + i * innerSize + inner;
                    sum += data[index];
                }

                int resultIndex = outer * innerSize + inner;
                resultData[resultIndex] = sum;
            }
        }

        return result;
    }

    @Override
    public Tensor mean(int dim, boolean keepDim) {
        Tensor summed = this.sum(dim, keepDim);

        float divisor = shape[dim];
        float[] resultData = summed.data().clone();

        for (int i = 0; i < resultData.length; i++) {
            resultData[i] /= divisor;
        }

        return Tensors.of(summed.shape(), resultData);
    }

    @Override
    public Tensor sign() {
        Tensor result = Tensors.create(shape);

        for (int i = 0; i < data.length; i++) {
            result.data()[i] = Math.signum(data[i]);
        }

        return result;
    }

    @Override
    public Tensor reshape(int... newShape) {
        int newSize = computeSize(newShape);

        if (newSize != data.length) {
            throw new IllegalArgumentException(
                    "The total new dimension (" + newSize + ") does not match the current dimension (" + data.length + ")"
            );
        }

        return Tensors.of(newShape, data);
    }


    @Override
    public Tensor view(int... newShape) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    @Override
    public Tensor permute(int... dims) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    @Override
    public Tensor squeeze() {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    @Override
    public Tensor squeeze(int dim) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    @Override
    public Tensor unsqueeze(int dim) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    @Override
    public Tensor select(int dim, int index) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    @Override
    public Tensor slice(Range... ranges) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    @Override
    public Tensor map(DoubleToDoubleFunction function) {
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) function.apply(data[i]);
        }

        return this;
    }

    @Override
    public Tensor fill(float value) {
        Arrays.fill(data, value);
        return this;
    }

    @Override
    public Tensor fill(Supplier<Double> supplier) {
        for (int i = 0; i < data.length; i++) {
            data[i] = supplier.get().floatValue();
        }

        return this;
    }

    @Override
    public AutogradContext autogradContext() {
        return autogradContext;
    }

    @Override
    public void setAutogradContext(AutogradContext context) {
        this.autogradContext = context;
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
        backward(Tensors.ones(shape));
    }

    @Override
    public void backward(Tensor gradOutput) {
        if (autogradContext == null) {
            throw new IllegalArgumentException("Autograd is not enabled for this tensor");
        }

        autogradContext.backward(gradOutput);
    }

    @Override
    public Tensor forward(Operation operation, Tensor other) {
        Tensor result = operation.forward(this, other);
        AutogradContext resultContext = result.autogradContext();

        if (resultContext == null) {
            resultContext = new AutogradContext(true);
        }

        resultContext.setOperation(operation, this, other);
        result.setAutogradContext(resultContext);

        return result;
    }

    @Override
    public Tensor addGrad(Tensor other) {
        if (!usesGrad() && !other.usesGrad()) {
            throw new IllegalArgumentException("Both tensors should be used with backflow!");
        }

        return forward(new AddOperation(), other);
    }

    @Override
    public Tensor mulGrad(Tensor other) {
        if (!usesGrad() && !other.usesGrad()) {
            throw new IllegalArgumentException("Both tensors should be used with backflow!");
        }

        return forward(new MulOperation(), other);
    }

    @Override
    public Tensor divGrad(Tensor other) {
        if (!usesGrad() && !other.usesGrad()) {
            throw new IllegalArgumentException("Both tensors should be used with backflow!");
        }

        return forward(new DivOperation(), other);
    }

    @Override
    public Tensor subGrad(Tensor other) {
        if (!usesGrad() && !other.usesGrad()) {
            throw new IllegalArgumentException("Both tensors should be used with backflow!");
        }

        return forward(new SubOperation(), other);
    }

    @Override
    public Tensor matmulGrad(Tensor other) {
        if (!usesGrad() && !other.usesGrad()) {
            throw new IllegalArgumentException("Both tensors should be used with backflow!");
        }

        return forward(new MatMulOperation(), other);
    }

    @Override
    public Tensor activateGrad(Activation activation) {
        if (!usesGrad()) {
            throw new IllegalArgumentException("Tensor does not use backflow!");
        }

        return forward(new ActivationOperation(activation), null);
    }

    @Override
    public Tensor transposeGrad() {
        if (!usesGrad()) {
            throw new IllegalArgumentException("Tensor does not use backflow!");
        }

        return forward(new TransposeOperation(), null);
    }

    @Override
    public Tensor convolve(Tensor kernel) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    @Override
    public Tensor softmax() {
        return softmax(1.0);
    }

    @Override
    public Tensor softmax(double temperature) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    @Override
    public String toString(String format) {
        if (shape.length == 0) {
            return format.formatted(data[0]);
        }

        StringBuilder result = new StringBuilder();
        appendTensor(result, 0, new int[shape.length], format);

        return result.toString();
    }

    @Override
    public String toString() {
        return toString("%.16f");
    }
}
