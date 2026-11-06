package org.brain4j.math;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;
import java.util.stream.IntStream;

import org.brain4j.math.commons.Range;
import org.brain4j.math.tensor.DType;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.engine.impl.JavaTensorEngine;
import org.brain4j.math.tensor.engine.TensorEngine;
import org.brain4j.math.tensor.impl.CpuTensor;

/**
 * Static utility methods for creating, initializing and manipulating {@link Tensor} instances.
 *
 * <p>This class acts as a central factory for common tensor constructions
 * (scalars, vectors, matrices, identity tensors, random initialization),
 * as well as low-level shape, indexing and broadcasting utilities.
 *
 * <p>Unless otherwise specified, all tensors produced by this class use the
 * currently configured tensor engine.
 *
 * <p>The methods provided here are intentionally low-level and performance-oriented;
 * most of them do not perform extensive validation beyond shape compatibility.
 * Incorrect usage may therefore result in undefined behavior or runtime exceptions.
 */
public final class Tensors {
    private static volatile TensorEngine engine = new JavaTensorEngine();

    private Tensors() {
    }

    public static TensorEngine engine() {
        return engine;
    }

    public static void setEngine(TensorEngine tensorEngine) {
        if (tensorEngine == null) {
            throw new NullPointerException("tensorEngine");
        }
        engine = tensorEngine;
    }

    /**
     * Creates a scalar {@link Tensor} containing a single value.
     *
     * <p>The resulting tensor has shape {@code [1]}.
     *
     * @param value the scalar value
     * @return a rank-1 tensor of length 1
     */
    public static Tensor scalar(double value) {
        return create(Shape.of(1), (float) value);
    }

    /**
     * Creates a tensor with the given shape and backing data buffer.
     *
     * <p>The data array is interpreted in row-major order and its length
     * must match the total number of elements implied by {@code shape}.
     *
     * @param shape the tensor shape
     * @param data the backing data buffer
     * @return a new {@link Tensor} instance
     */
    public static Tensor create(int[] shape, float... data) {
        return create(Shape.of(shape), data);
    }

    /**
     * Creates a tensor with the given shape and backing data buffer.
     *
     * <p>The data array is interpreted in row-major order and its length
     * must match the total number of elements implied by {@code shape}.
     *
     * @param shape the tensor shape
     * @param data the backing data buffer
     * @return a new {@link Tensor} instance
     */
    public static Tensor create(Shape shape, float... data) {
        return create(shape, DType.FLOAT32, data);
    }

    public static Tensor create(Shape shape, DType dtype, float... data) {
        return engine.create(shape, dtype, data);
    }

    public static Tensor create(TensorEngine tensorEngine, Shape shape, DType dtype, float... data) {
        if (tensorEngine == null) {
            throw new NullPointerException("tensorEngine");
        }
        return tensorEngine.create(shape, dtype, data);
    }

    /**
     * Creates a tensor with the given shape, explicit strides and backing data buffer.
     *
     * <p>This constructor allows full control over the tensor memory layout.
     * The {@code strides} array defines how indices are mapped to the underlying
     * flat {@code data} buffer.
     *
     * <p>No validation is performed to ensure consistency between shape,
     * strides and data length.
     *
     * @param shape the tensor shape
     * @param strides the explicit strides
     * @param data the backing data buffer
     * @return a new {@link Tensor} instance
     */
    public static Tensor create(int[] shape, int[] strides, float... data) {
        return new CpuTensor(Shape.of(shape), strides, data);
    }

    /**
     * Creates a 1D tensor (vector) from the provided data.
     * <p>The resulting tensor has shape {@code [data.length]}.
     *
     * @param data vector elements
     * @return a 1D tensor
     */
    public static Tensor vector(float... data) {
        return create(Shape.of(data.length), data);
    }

    /**
     * Creates a 2D tensor (matrix) with the given dimensions and data.
     *
     * <p>The data array is interpreted in row-major order and must contain
     * exactly {@code rows * cols} elements.
     *
     * @param rows number of rows
     * @param cols number of columns
     * @param data matrix elements in row-major order
     * @return a 2D tensor
     */
    public static Tensor matrix(int rows, int cols, float... data) {
        return create(Shape.of(rows, cols), data);
    }

    /**
     * Returns a tensor filled with zeros of the requested shape.
     *
     * @param shape desired shape
     * @return a tensor initialized with zeros
     */
    public static Tensor zeros(int... shape) {
        return zeros(Shape.of(shape));
    }

    /**
     * Returns a tensor filled with zeros of the requested shape.
     *
     * @param shape desired shape
     * @return a tensor initialized with zeros
     */
    public static Tensor zeros(Shape shape) {
        return engine.zeros(shape, DType.FLOAT32);
    }

    // TODO: javadoc
    public static Tensor zeros(Shape shape, DType dtype) {
        return engine.zeros(shape, dtype);
    }

    /**
     * Creates a 1D tensor containing a contiguous range of integer values.
     *
     * <p>The resulting tensor has shape {@code [end - start]} and contains
     * values {@code start, start + 1, ..., end - 1}.
     *
     * @param start inclusive start value
     * @param end   exclusive end value
     * @return a range tensor
     */
    public static Tensor range(int start, int end) {
        int length = end - start;
        Tensor result = Tensors.zeros(length);

        for (int i = 0; i < length; i++) {
            result.data()[i] = i + start;
        }

        return result;
    }

    /**
     * Returns a tensor filled with ones of the requested shape.
     *
     * @param shape desired shape
     * @return a tensor initialized with ones
     */
    public static Tensor ones(int... shape) {
        return ones(Shape.of(shape));
    }

    /**
     * Returns a tensor filled with ones of the requested shape.
     *
     * @param shape desired shape
     * @return a tensor initialized with ones
     */
    public static Tensor ones(Shape shape) {
        Tensor result = zeros(shape);
        Arrays.fill(result.data(), 1);
        return result;
    }

    /**
     * Creates a tensor filled with uniformly distributed random values
     * in the range {@code [0, 1)}.
     *
     * @param generator the random number generator to use
     * @param shape     the desired tensor shape
     * @return a randomly initialized tensor
     */
    public static Tensor random(RandomGenerator generator, int... shape) {
        Tensor result = Tensors.zeros(shape);
        return result.map(x -> generator.nextFloat());
    }

    /**
     * Creates a tensor filled with uniformly distributed random values
     * in the range {@code [0, 1)}.
     *
     * @param generator the random number generator to use
     * @param shape     the desired tensor shape
     * @return a randomly initialized tensor
     */
    public static Tensor random(RandomGenerator generator, Shape shape) {
        Tensor result = Tensors.zeros(shape);
        return result.map(x -> generator.nextFloat());
    }

    /**
     * Creates a tensor filled with uniformly distributed random values
     * in the range {@code [0, 1)}.
     * <p>By default, uses a new instance of {@link SplittableRandom}.
     *
     * @param shape     the desired tensor shape
     * @return a randomly initialized tensor
     */
    public static Tensor random(int... shape) {
        return random(new SplittableRandom(), shape);
    }

    /**
     * Creates a tensor filled with uniformly distributed random values
     * in the range {@code [0, 1)}.
     * <p>By default, uses a new instance of {@link SplittableRandom}.
     *
     * @param shape     the desired tensor shape
     * @return a randomly initialized tensor
     */
    public static Tensor random(Shape shape) {
        return random(new SplittableRandom(), shape);
    }

    /**
     * Merges a list of tensors into a single batched tensor.
     *
     * <p>Each input tensor is first expanded with a leading dimension
     * (via {@code unsqueeze}), then concatenated along dimension 0.
     *
     * <p>All tensors must have identical shapes.
     *
     * @param tensors list of tensors to merge
     * @return a batched tensor
     * @throws IllegalArgumentException if the list is empty
     */
    public static Tensor mergeTensors(List<Tensor> tensors) {
        if (tensors.isEmpty()) {
            throw new IllegalArgumentException("No tensors provided!");
        }

        Tensor result = tensors.getFirst().unsqueeze();

        for (int i = 1; i < tensors.size(); i++) {
            Tensor current = tensors.get(i).unsqueeze();
            result = result.concat(current, 0);
        }

        return result;
    }

    /**
     * Creates a triangular mask tensor with negative infinity above the main diagonal.
     *
     * <p>The mask contains {@link Float#NEGATIVE_INFINITY} for positions
     * where {@code column > row}, and zero elsewhere. This is typically used
     * for causal attention masking in transformer models.
     *
     * @param firstDim number of rows
     * @param secondDim number of columns
     * @return a triangular mask tensor
     */
    public static Tensor triangularMask(int firstDim, int secondDim) {
        Tensor mask = Tensors.zeros(firstDim, secondDim);

        for (int i = 0; i < firstDim; i++) {
            for (int j = i + 1; j < secondDim; j++) {
                mask.set(Float.NEGATIVE_INFINITY, i, j);
            }
        }

        return mask;
    }

    /**
     * Delegates to {@link Tensors#triangularMask(int, int)} with first and seecond dimensions equal.
     * @param dimension number of rows and columns
     * @return a triangular mask tensor
     */
    public static Tensor triangularMask(int dimension) {
        return triangularMask(dimension, dimension);
    }

    /**
     * Concatenates the given tensors along the last dimension.
     *
     * @param tensors tensors to concatenate
     * @return the concatenated tensor
     */
    public static Tensor concat(Tensor... tensors) {
        return concat(List.of(tensors), -1);
    }

    /**
     * Concatenates multiple tensors along a given dimension.
     *
     * <p>If {@code dim == -1}, concatenation is performed along the last dimension.
     *
     * @param tensors tensors to concatenate
     * @param dim     dimension along which to concatenate
     * @return the concatenated tensor
     */
    public static Tensor concat(List<Tensor> tensors, int dim) {
        Tensor base = tensors.getFirst();

        for (int i = 1; i < tensors.size(); i++) {
            base = base.concat(tensors.get(i), dim);
        }

        return base;
    }

    /**
     * Concatenates tensors along the last dimension while preserving
     * gradient propagation semantics and autograd context.
     *
     * @param tensors tensors to concatenate
     * @return the concatenated tensor with gradient support
     */
    public static Tensor concatGrad(List<Tensor> tensors) {
        return concatGrad(tensors, -1);
    }

    /**
     * Concatenates tensors along a given dimension while preserving
     * gradient propagation semantics and autograd context.
     *
     * @param tensors tensors to concatenate
     * @param dim concatenation dimension
     * @return the concatenated tensor with gradient support
     */
    public static Tensor concatGrad(List<Tensor> tensors, int dim) {
        Tensor base = tensors.getFirst();

        for (int i = 1; i < tensors.size(); i++) {
            base = base.concat(tensors.get(i), dim);
        }

        return base;
    }

    /**
     * Generates a matrix with approximately orthonormal columns using
     * the Gram–Schmidt process.
     *
     * <p>The resulting tensor has shape {@code [rows, cols]}.
     * <p>This is useful when initializing weights in recurrent neural networks.
     *
     * @param rows number of rows
     * @param cols number of columns
     * @return an orthogonal matrix
     */
    public static Tensor orthogonal(int rows, int cols) {
        Random rng = new Random();
        Tensor A = Tensors.zeros(rows, cols).map(x -> (float) rng.nextGaussian());

        List<Tensor> Q = new ArrayList<>();

        for (int i = 0; i < cols; i++) {
            Tensor ai = A.slice(Range.all(), Range.point(i));
            Tensor ui = ai;

            for (Tensor qj : Q) {
                Tensor dot = qj.transpose().matmul(ai); // [1, 1]
                ui = ui.minus(qj.times(dot.get(0, 0)));
            }

            double norm = Math.sqrt(ui.copy().pow(2).sum());
            Tensor qi = ui.divide(norm);

            Q.add(qi);
        }

        return Tensors.concat(Q, 1);
    }

    /**
     * Creates an identity matrix of size {@code n x n}.
     *
     * @param n matrix dimension
     * @return an identity tensor
     */
    public static Tensor eye(int n) {
        Tensor result = Tensors.zeros(n, n);

        for (int i = 0; i < n; i++) {
            result.set(1, i, i);
        }

        return result;
    }

    /**
     * Creates a zero-filled tensor with the same shape as the given tensor.
     *
     * @param other reference tensor
     * @return a zero-initialized tensor with identical shape
     */
    public static Tensor zerosLike(Tensor other) {
        return Tensors.zeros(other.shape());
    }


    /**
     * Transforms an input image tensor into column format (im2col) with a configurable stride.
     *
     * @param input input image tensor
     * @param filterHeight convolution kernel height
     * @param filterWidth convolution kernel width
     * @param stride stride to apply on both spatial dimensions
     * @return a 2D tensor of shape {@code [patchSize, totalPatches]}
     */
    public static Tensor im2col(Tensor input, int filterHeight, int filterWidth, int stride) {
        int[] shape = input.shape();
        int channels = shape[0];
        int inHeight = shape[1];
        int inWidth = shape[2];

        int outHeight = (inHeight - filterHeight) / stride + 1;
        int outWidth = (inWidth - filterWidth) / stride + 1;

        int patchSize = channels * filterHeight * filterWidth;
        int totalPatches = outHeight * outWidth;

        float[] inputData = input.data();
        float[] resultData = new float[patchSize * totalPatches];

        // TODO: delete this
        return Tensors.create(Shape.of(patchSize, totalPatches), resultData);
    }

    /**
     * Returns the indices of the {@code topK} largest values in the array.
     *
     * <p>The returned indices are sorted in descending order of their
     * corresponding values.
     *
     * @param topK number of top elements to select
     * @param data input data array
     * @return indices of the top-K elements
     */
    public static int[] topK(int topK, float[] data) {
        return IntStream.range(0, data.length)
            .boxed()
            .sorted((i, j) -> Double.compare(data[j], data[i]))
            .limit(topK)
            .mapToInt(Integer::intValue)
            .toArray();
    }

    /**
     * Broadcasts two shapes following NumPy rules.
     *
     * @return the broadcast shape
     * @throws IllegalArgumentException if shapes are incompatible
     */
    public static int[] broadcastShapes(int[] a, int[] b) {
        int len = Math.max(a.length, b.length);
        int[] result = new int[len];

        for (int i = 0; i < len; i++) {
            int ai = i >= len - a.length ? a[i - (len - a.length)] : 1;
            int bi = i >= len - b.length ? b[i - (len - b.length)] : 1;

            if (ai != bi && ai != 1 && bi != 1) {
                throw new IllegalArgumentException("Incompatible dimensions for broadcasting");
            }

            result[i] = Math.max(ai, bi);
        }

        return result;
    }

    /**
     * Maps an output index back to a broadcast input index.
     *
     * @return the index within the target shape
     */
    public static int[] broadcastIndex(int[] outIdx, int[] outShape, int[] targetShape) {
        int offset = outShape.length - targetShape.length;
        int[] result = new int[targetShape.length];

        for (int i = 0; i < targetShape.length; i++) {
            result[i] = targetShape[i] == 1 ? 0 : outIdx[i + offset];
        }

        return result;
    }

    /**
     * Flattens a multidimensional index using the given strides.
     *
     * @return the linear index
     */
    public static int flattenIndex(int[] idx, int[] strides) {
        int sum = 0;

        for (int i = 0; i < idx.length; i++) {
            sum += idx[i] * strides[i];
        }

        return sum;
    }

    /**
     * Computes row-major strides for the given shape.
     *
     * @return the stride array
     */
    public static int[] computeStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int prod = 1;

        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = prod;
            prod *= shape[i];
        }

        return strides;
    }

    /**
     * Converts a linear index into multidimensional indices.
     *
     * @return the indices within the given shape
     */
    public static int[] unravelIndex(int linearIndex, int[] shape) {
        int[] indices = new int[shape.length];

        for (int i = shape.length - 1; i >= 0; i--) {
            indices[i] = linearIndex % shape[i];
            linearIndex /= shape[i];
        }

        return indices;
    }

    /**
     * Returns the element count for the given shape.
     *
     * @return the total number of elements
     */
    public static int computeSize(int[] shape) {
        int size = 1;

        for (int dim : shape) {
            size *= dim;
        }

        return size;
    }

    /**
     * Returns the shape left after reducing one dimension.
     *
     * @param dim dimension to reduce, negative counts from the end
     * @param keepDim true to keep the dimension with size 1
     * @return the new shape
     */
    public static int[] computeNewShape(int[] shape, int dim, boolean keepDim) {
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
}
