package org.brain4j.math;

import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.convolution.im2col.Im2ColParams;
import org.brain4j.math.convolution.im2col.Im2ColTask;
import org.brain4j.math.tensor.impl.CpuTensor;
import org.brain4j.math.commons.Range;
import org.brain4j.math.operations.ParallelConvolve;

import java.util.*;
import java.util.concurrent.ForkJoinPool;
import java.util.random.RandomGenerator;
import java.util.stream.IntStream;

/**
 * Static utility methods for creating, initializing and manipulating {@link Tensor} instances.
 *
 * <p>This class acts as a central factory for common tensor constructions
 * (scalars, vectors, matrices, identity tensors, random initialization),
 * as well as low-level shape, indexing and broadcasting utilities.
 *
 * <p>Unless otherwise specified, all tensors produced by this class are
 * CPU-backed and implemented as {@link CpuTensor}.
 *
 * <p>The methods provided here are intentionally low-level and performance-oriented;
 * most of them do not perform extensive validation beyond shape compatibility.
 * Incorrect usage may therefore result in undefined behavior or runtime exceptions.
 */
public class Tensors {

    /** Number of threads suggested for parallel operations. */
    public static final int PARALLELISM = Runtime.getRuntime().availableProcessors();

    /** Complexity threshold used to decide when to split tasks for parallel algorithms. */
    public static final int SPLIT_COMPLEXITY_THRESHOLD = 1 << 10; // 1024

    /**
     * Creates a scalar {@link Tensor} containing a single value.
     *
     * <p>The resulting tensor has shape {@code [1]}.
     *
     * @param value the scalar value
     * @return a rank-1 tensor of length 1
     */
    public static Tensor scalar(double value) {
        return new CpuTensor(Shape.of(1), (float) value);
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
        return new CpuTensor(shape, data);
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
        return new CpuTensor(Shape.of(shape));
    }

    /**
     * Returns a tensor filled with zeros of the requested shape.
     *
     * @param shape desired shape
     * @return a tensor initialized with zeros
     */
    public static Tensor zeros(Shape shape) {
        return new CpuTensor(shape);
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
        Tensor result = new CpuTensor(shape);
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
     * Performs a convolution between an input tensor and a kernel tensor.
     *
     * <p>This method delegates to {@link ParallelConvolve} and may execute
     * in parallel depending on tensor size.
     *
     * @param input input tensor
     * @param kernel convolution kernel
     * @return the result of the convolution
     */
    public static Tensor convolve(Tensor input, Tensor kernel) {
        return convolve(input, kernel, 1);
    }

    /**
     * Performs a convolution between an input tensor and a kernel tensor with a configurable stride.
     *
     * @param input input tensor
     * @param kernel convolution kernel
     * @param stride stride to apply on both spatial dimensions
     * @return the result of the convolution
     */
    public static Tensor convolve(Tensor input, Tensor kernel, int stride) {
        return ParallelConvolve.convolve(input, kernel, stride);
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
            base = base.concatGrad(tensors.get(i), dim);
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
        Tensor A = Tensors.zeros(rows, cols).map(x -> rng.nextGaussian());

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
     * Transforms an input image tensor into column format (im2col),
     * typically used to express convolution as a matrix multiplication.
     *
     * <p>The input tensor is expected to have shape {@code [channels, height, width]}.
     *
     * @param input        input image tensor
     * @param filterHeight convolution kernel height
     * @param filterWidth  convolution kernel width
     * @return a 2D tensor of shape {@code [patchSize, totalPatches]}
     */
    public static Tensor im2col(Tensor input, int filterHeight, int filterWidth) {
        return im2col(input, filterHeight, filterWidth, 1);
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

        Im2ColParams params = new Im2ColParams(
            inputData,
            resultData,
            0,
            0,
            channels,
            inHeight,
            inWidth,
            filterHeight,
            filterWidth,
            outHeight,
            outWidth
        );

        try (ForkJoinPool pool = ForkJoinPool.commonPool()) {
            pool.invoke(new Im2ColTask(params, 0, totalPatches));
        }

        return Tensors.create(Shape.of(patchSize, totalPatches), resultData);
    }

    /**
     * Reconstructs an image tensor from its column representation (col2im).
     *
     * <p>This operation accumulates overlapping patches.
     *
     * @param cols         column tensor
     * @param channels     number of channels
     * @param inHeight     input height
     * @param inWidth      input width
     * @param filterHeight kernel height
     * @param filterWidth  kernel width
     * @return reconstructed image tensor
     */
    public static Tensor col2im(Tensor cols, int channels, int inHeight, int inWidth,
                                int filterHeight, int filterWidth) {
        return col2im(cols, channels, inHeight, inWidth, filterHeight, filterWidth, 1);
    }

    /**
     * Reconstructs an image tensor from its column representation (col2im) with a configurable stride.
     *
     * @param cols column tensor
     * @param channels number of channels
     * @param inHeight input height
     * @param inWidth input width
     * @param filterHeight kernel height
     * @param filterWidth kernel width
     * @param stride stride used during im2col
     * @return reconstructed image tensor
     */
    public static Tensor col2im(Tensor cols, int channels, int inHeight, int inWidth,
                                int filterHeight, int filterWidth, int stride) {
        int[] colShape = cols.shape(); // [patchSize, totalPatches]
        int patchSize = colShape[0];
        int totalPatches = colShape[1];

        int outWidth = (inWidth - filterWidth) / stride + 1;

        float[] colData = cols.data();
        float[] imgData = new float[channels * inHeight * inWidth];

        for (int patchIdx = 0; patchIdx < totalPatches; patchIdx++) {
            int outRow = patchIdx / outWidth;
            int outCol = patchIdx % outWidth;
            int inRow = outRow * stride;
            int inCol = outCol * stride;
            int baseOffset = patchIdx * patchSize;

            for (int c = 0; c < channels; c++) {
                int channelOffset = c * filterHeight * filterWidth;
                int imgChannelOffset = c * inHeight * inWidth;

                for (int fh = 0; fh < filterHeight; fh++) {
                    int srcPos = baseOffset + channelOffset + fh * filterWidth;
                    int destPos = imgChannelOffset + (inRow + fh) * inWidth + inCol;
                    for (int fw = 0; fw < filterWidth; fw++) {
                        imgData[destPos + fw] += colData[srcPos + fw];
                    }
                }
            }
        }

        return Tensors.create(Shape.of(channels, inHeight, inWidth), imgData);
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

    public static int[] broadcastShapes(int[] a, int[] b) {
        int len = Math.max(a.length, b.length);
        int[] result = new int[len];
        for (int i = 0; i < len; i++) {
            int ai = i >= len - a.length ? a[i - (len - a.length)] : 1;
            int bi = i >= len - b.length ? b[i - (len - b.length)] : 1;
            if (ai != bi && ai != 1 && bi != 1)
                throw new IllegalArgumentException("Incompatible dimensions for broadcasting");
            result[i] = Math.max(ai, bi);
        }
        return result;
    }

    public static int[] broadcastIndex(int[] outIdx, int[] outShape, int[] targetShape) {
        int offset = outShape.length - targetShape.length;
        int[] result = new int[targetShape.length];

        for (int i = 0; i < targetShape.length; i++) {
            result[i] = targetShape[i] == 1 ? 0 : outIdx[i + offset];
        }

        return result;
    }
    
    public static int flattenIndex(int[] idx, int[] strides) {
        int sum = 0;

        for (int i = 0; i < idx.length; i++) {
            sum += idx[i] * strides[i];
        }

        return sum;
    }
    
    public static int[] computeStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int prod = 1;

        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = prod;
            prod *= shape[i];
        }

        return strides;
    }
    
    public static int[] unravelIndex(int linearIndex, int[] shape) {
        int[] indices = new int[shape.length];

        for (int i = shape.length - 1; i >= 0; i--) {
            indices[i] = linearIndex % shape[i];
            linearIndex /= shape[i];
        }

        return indices;
    }

    public static int computeSize(int[] shape) {
        int size = 1;

        for (int dim : shape) {
            size *= dim;
        }

        return size;
    }

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
