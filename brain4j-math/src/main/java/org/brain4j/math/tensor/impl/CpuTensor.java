package org.brain4j.math.tensor.impl;

import java.nio.ByteBuffer;
import java.util.function.DoubleSupplier;
import java.util.function.IntConsumer;

import org.brain4j.math.Tensors;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.commons.FloatToFloatFunction;
import org.brain4j.math.commons.Range;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.operations.AutogradContext;
import org.brain4j.math.operations.Operation;
import org.brain4j.math.tensor.DType;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.TensorLayout;
import org.brain4j.math.tensor.storage.impl.CpuStorage;

// TODO
public record CpuTensor(CpuStorage storage, TensorLayout layout) implements Tensor {

    public CpuTensor(Shape shape) {
        this(shape, new float[shape.size()]);
    }

    public CpuTensor(Shape shape, float... data) {
        this(shape, Tensors.computeStrides(shape.dims()), data);
    }

    public CpuTensor(Shape shape, int[] strides, float... data) {
        this(new CpuStorage(DType.FLOAT32, data),
                new TensorLayout(shape.dims(), strides, 0));
        validateStorageBounds();
    }

    private void validateStorageBounds() {
        if (layout.elements() == 0) {
            return;
        }
        int maximum = layout.offset();
        for (int i = 0; i < layout.rank(); i++) {
            maximum += (layout.shape()[i] - 1) * layout.strides()[i];
        }
        if (layout.offset() < 0 || maximum >= storage.elements()) {
            throw new IllegalArgumentException("Layout exceeds storage bounds");
        }
    }

    private CpuTensor view(TensorLayout viewLayout) {
        return new CpuTensor(storage, viewLayout);
    }

    public CpuStorage storage() {
        return storage;
    }

    public TensorLayout layout() {
        return layout;
    }

    private void forEachIndex(IntConsumer consumer) {
        int count = elements();
        int physicalIndex = layout.offset();

        if (layout.isContiguous()) {
            for (int linear = 0; linear < count; linear++) {
                consumer.accept(physicalIndex + linear);
            }
            return;
        }

        int[] shape = layout.shape();
        int[] strides = layout.strides();
        int[] indices = new int[rank()];

        for (int linear = 0; linear < count; linear++) {
            consumer.accept(physicalIndex);
            if (linear + 1 < count) {
                physicalIndex = CpuStorage.advanceIndex(indices, shape, strides, physicalIndex);
            }
        }
    }

    @Override
    public int shapeAt(int index) {
        return layout.shape()[index];
    }

    @Override
    public int[] shape() {
        return layout.shape();
    }

    @Override
    public float[] data() {
        if (!layout.isContiguous() || layout.offset() != 0) {
            return storage.copy(layout).data();
        }
        return storage.data();
    }

    @Override
    public int[] strides() {
        return layout.strides();
    }

    @Override
    public DType dtype() {
        return storage.dtype();
    }

    @Override
    public boolean isContiguous() {
        return layout.isContiguous() && layout.offset() == 0;
    }

    @Override
    public byte[] toByteArray() {
        byte[] result = new byte[elements() * Float.BYTES];
        ByteBuffer buffer = ByteBuffer.wrap(result);
        forEachIndex(index -> buffer.putFloat(storage.getFloat(index)));
        return result;
    }

    @Override
    public int linearIndex(int... indices) {
        return layout.linearIndex(indices);
    }

    @Override
    public float get(int... indices) {
        return storage.getFloat(linearIndex(indices));
    }

    @Override
    public Tensor set(float value, int... indices) {
        storage.setFloat(linearIndex(indices), value);
        return this;
    }

    @Override
    public int rank() {
        return layout.rank();
    }

    @Override
    public int elements() {
        return layout.elements();
    }

    @Override
    public int argmax() {
        if (elements() == 0) {
            throw new IllegalStateException("Cannot compute argmax of an empty tensor");
        }

        if (layout.isContiguous()) {
            return argmaxContiguous();
        }

        return argmaxStrided();
    }

    private int argmaxContiguous() {
        float[] values = storage.data();
        int offset = layout.offset();
        int best = 0;
        float maximum = values[offset];
        for (int linear = 1; linear < elements(); linear++) {
            float value = values[offset + linear];
            if (value > maximum) {
                maximum = value;
                best = linear;
            }
        }
        return best;
    }

    private int argmaxStrided() {
        int[] shape = layout.shape();
        int[] strides = layout.strides();
        int[] indices = new int[rank()];
        int physicalIndex = layout.offset();
        int best = 0;
        float maximum = storage.getFloat(physicalIndex);

        for (int linear = 1; linear < elements(); linear++) {
            physicalIndex = CpuStorage.advanceIndex(indices, shape, strides, physicalIndex);
            float value = storage.getFloat(physicalIndex);
            if (value > maximum) {
                maximum = value;
                best = linear;
            }
        }
        return best;
    }

    @Override
    public Tensor to(Device device) {
        return null;
    }

    @Override
    public Tensor add(Tensor other) {
        return null;
    }

    @Override
    public Tensor add(double value) {
        return null;
    }

    @Override
    public Tensor sub(Tensor other) {
        return null;
    }

    @Override
    public Tensor sub(double value) {
        return null;
    }

    @Override
    public Tensor mul(Tensor other) {
        return null;
    }

    @Override
    public Tensor mul(double value) {
        return null;
    }

    @Override
    public Tensor div(Tensor other) {
        return null;
    }

    @Override
    public Tensor div(double value) {
        return null;
    }

    @Override
    public Tensor pow(double value) {
        return null;
    }

    @Override
    public Tensor pow(Tensor other) {
        return null;
    }

    @Override
    public Tensor sqrt() {
        return null;
    }

    @Override
    public Tensor flatten() {
        return reshape(elements());
    }

    @Override
    public Tensor matmul(Tensor other) {
        return null;
    }

    @Override
    public Tensor convolve(Tensor kernel) {
        return null;
    }

    @Override
    public Tensor convolve(Tensor kernel, int stride) {
        return null;
    }

    @Override
    public double distance(Tensor other) {
        return 0;
    }

    @Override
    public double distanceSquared(Tensor other) {
        return 0;
    }

    @Override
    public Tensor broadcast(int[] targetShape) {
        return null;
    }

    @Override
    public Tensor broadcastLike(Tensor other) {
        return null;
    }

    @Override
    public double sum() {
        return 0;
    }

    @Override
    public double mean() {
        return 0;
    }

    @Override
    public double variance() {
        return 0;
    }

    @Override
    public double max() {
        return 0;
    }

    @Override
    public double min() {
        return 0;
    }

    @Override
    public Tensor sum(int dim, boolean keepDim) {
        return null;
    }

    @Override
    public Tensor mean(int dim, boolean keepDim) {
        return null;
    }

    @Override
    public Tensor variance(int dim, boolean keepDim) {
        return null;
    }

    @Override
    public Tensor variance(Tensor mean, int dim, boolean keepDim) {
        return null;
    }

    @Override
    public Tensor sign() {
        return null;
    }

    @Override
    public Tensor reshape(int... newShape) {
        try {
            return view(Shape.of(newShape));
        } catch (IllegalArgumentException exception) {
            return contiguous().view(Shape.of(newShape));
        }
    }

    @Override
    public Tensor view(Shape shape) {
        int[] newShape = shape.dims();
        int[] newStrides = calculateViewStrides(newShape);
        return view(new TensorLayout(newShape, newStrides, layout.offset()));
    }

    private int[] calculateViewStrides(int[] newShape) {
        if (Tensors.computeSize(newShape) != elements()) {
            throw new IllegalArgumentException("View changes element count");
        }

        int rank = rank();
        int[] oldShape = layout.shape();
        int[] oldStrides = layout.strides();
        int[] chunkSizes = new int[rank];
        int[] chunkStrides = new int[rank];
        int chunkCount = 0;

        for (int dimension = rank - 1; dimension >= 0; dimension--) {
            int size = oldShape[dimension];
            int stride = oldStrides[dimension];
            if (chunkCount == 0
                    || stride != chunkStrides[chunkCount - 1] * chunkSizes[chunkCount - 1]) {
                chunkSizes[chunkCount] = size;
                chunkStrides[chunkCount] = stride;
                chunkCount++;
            } else {
                chunkSizes[chunkCount - 1] *= size;
            }
        }

        int[] newStrides = new int[newShape.length];
        int chunk = 0;
        int remaining = chunkCount == 0 ? 1 : chunkSizes[chunkCount - 1];
        int stride = chunkCount == 0 ? 1 : chunkStrides[chunkCount - 1];
        int consumed = 1;

        for (int dimension = newShape.length - 1; dimension >= 0; dimension--) {
            int size = newShape[dimension];
            if (size < 0 || (size > 1 && remaining % size != 0)) {
                throw new IllegalArgumentException("Shape is incompatible with this tensor layout");
            }
            newStrides[dimension] = stride * consumed;
            if (size > 1) {
                remaining /= size;
                consumed *= size;
                if (remaining == 1 && chunk + 1 < chunkCount) {
                    chunk++;
                    remaining = chunkSizes[chunkCount - 1 - chunk];
                    stride = chunkStrides[chunkCount - 1 - chunk];
                    consumed = 1;
                }
            }
        }

        if (chunk + 1 < chunkCount || remaining != 1) {
            throw new IllegalArgumentException("Shape is incompatible with this tensor layout");
        }
        return newStrides;
    }

    @Override
    public Tensor contiguous() {
        if (layout.isContiguous() && layout.offset() == 0) {
            return this;
        }

        int[] strides = Tensors.computeStrides(shape());

        CpuStorage contiguousStorage = storage.copy(layout);
        TensorLayout contiguousLayout = new TensorLayout(shape(), strides, 0);

        return new CpuTensor(contiguousStorage, contiguousLayout);
    }

    @Override
    public Tensor squeeze() {
        return null;
    }

    @Override
    public Tensor squeeze(int dimension) {
        return null;
    }

    @Override
    public Tensor unsqueeze() {
        return null;
    }

    @Override
    public Tensor unsqueeze(int dimension) {
        return null;
    }

    @Override
    public Tensor transpose() {
        if (rank() < 2) {
            return this;
        }
        return transpose(rank() - 2, rank() - 1);
    }

    @Override
    public Tensor transpose(int dim1, int dim2) {
        // Handle negative dimensions
        int first = Math.floorMod(dim1, rank());
        int second = Math.floorMod(dim2, rank());
        int[] shape = layout.shape();
        int[] strides = layout.strides();

        int dimension = shape[first];

        shape[first] = shape[second];
        shape[second] = dimension;

        dimension = strides[first];
        strides[first] = strides[second];
        strides[second] = dimension;

        return view(new TensorLayout(shape, strides, layout.offset()));
    }

    @Override
    public boolean transposed() {
        return !layout.isContiguous();
    }

    @Override
    public Tensor concat(Tensor other) {
        return null;
    }

    @Override
    public Tensor concat(Tensor other, int dimension) {
        return null;
    }

    @Override
    public Tensor activate(Activation activation) {
        return null;
    }

    @Override
    public Tensor select(int dim, int index) {
        return null;
    }

    @Override
    public Tensor slice(Range... ranges) {
        return null;
    }

    @Override
    public Tensor map(FloatToFloatFunction function) {
        forEachIndex(index -> storage.setFloat(index, function.apply(storage.getFloat(index))));
        return this;
    }

    @Override
    public Tensor fill(float value) {
        forEachIndex(index -> storage.setFloat(index, value));
        return this;
    }

    @Override
    public Tensor fill(DoubleSupplier supplier) {
        return null;
    }

    @Override
    public AutogradContext autogradContext() {
        return null;
    }

    @Override
    public void setAutogradContext(AutogradContext autogradContext) {

    }

    @Override
    public Tensor withGrad() {
        return null;
    }

    @Override
    public Tensor noGrad() {
        return null;
    }

    @Override
    public boolean usesGrad() {
        return false;
    }

    @Override
    public void zeroGrad() {

    }

    @Override
    public Tensor grad() {
        return null;
    }

    @Override
    public void backward() {

    }

    @Override
    public void backward(Tensor gradOutput) {

    }

    @Override
    public Tensor forward(Operation operation) {
        return null;
    }

    @Override
    public Tensor forward(Operation operation, Tensor other) {
        return null;
    }

    @Override
    public Tensor forward(Operation operation, Tensor... others) {
        return null;
    }

    @Override
    public Tensor softmax() {
        return null;
    }

    @Override
    public Tensor softmax(double temperature) {
        return null;
    }

    @Override
    public String toString(String format) {
        return "";
    }

    @Override
    public Tensor copy() {
        CpuStorage copiedStorage = storage.copy(layout);
        TensorLayout copiedLayout = new TensorLayout(shape(), Tensors.computeStrides(shape()), 0);

        return new CpuTensor(copiedStorage, copiedLayout);
    }
}
