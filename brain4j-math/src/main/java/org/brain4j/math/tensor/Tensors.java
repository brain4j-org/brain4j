package org.brain4j.math.tensor;

import org.brain4j.math.tensor.impl.cpu.CpuTensor;

import java.util.Arrays;
import java.util.List;

public class Tensors {

    public static Tensor scalar(double value) {
        return new CpuTensor(new int[]{1}, (float) value);
    }

    public static Tensor create(int[] shape, float... data) {
        return new CpuTensor(shape, data);
    }
    
    public static Tensor vector(float... data) {
        return create(new int[]{data.length}, data);
    }

    public static Tensor matrix(int rows, int cols, float... data) {
        return create(new int[]{rows, cols}, data);
    }
    
    public static Tensor zeros(int... shape) {
        return new CpuTensor(shape);
    }
    
    public static Tensor ones(int... shape) {
        Tensor result = new CpuTensor(shape);
        Arrays.fill(result.data(), 1);
        return result;
    }

    public static Tensor random(int...  shape) {
        Tensor result = Tensors.zeros(shape);
        return result.map(x -> Math.random() * 2 - 1);
    }

    public static Tensor concat(List<Tensor> inputs) {
        Tensor base = inputs.getFirst();

        int[] baseShape = base.shape();
        int totalColumns = 0;
        int rank = baseShape.length;;

        for (Tensor input : inputs) {
            int[] shape = input.shape();

            if (shape.length != rank) {
                throw new IllegalArgumentException(
                    "All tensors must have the same dimension!"
                );
            }

            for (int i = 0; i < shape.length - 1; i++) {
                if (shape[i] != baseShape[i]) {
                    throw new IllegalArgumentException(
                        "All tensors must have the same base shape! [" + Arrays.toString(input.shape()) + "]"
                    );
                }
            }

            totalColumns += shape[shape.length - 1];
        }

        int[] paddedShape = new int[rank];

        System.arraycopy(baseShape, 0, paddedShape, 0, paddedShape.length);
        paddedShape[paddedShape.length - 1] = totalColumns;

        Tensor result = zeros(paddedShape);

        int offset = 0;

        for (Tensor input : inputs) {
            int size = input.shape()[rank - 1];

            result.setSliceAlongLastDim(offset, input);

            offset += size;
        }

        return result;
    }

    /**
     * Stacks two matrices into a single tensor.
     * @apiNote The first shape can be different, meanwhile the second shape must be equal for all tensors
     * @param tensors The input tensors
     * @return A new tensor containing all the other ones one on top of the other
     */
    public static Tensor stack(Tensor... tensors) {
        if (tensors.length == 0) {
            throw new IllegalArgumentException("No elements specified!");
        }

        int dimension = Integer.MAX_VALUE;
        int elements = 0;

        for (Tensor tensor : tensors) {
            if (tensor.dimension() != 2) {
                throw new IllegalArgumentException("All tensors must be 2 dimensional!");
            }

            int[] shape = tensor.shape();

            if (dimension == Integer.MAX_VALUE) {
                dimension = shape[1];
            } else if (dimension != shape[1]) {
                throw new IllegalArgumentException("Not all tensors have the same dimension!");
            }

            elements += shape[0];
        }

        Tensor result = Tensors.zeros(elements, dimension);
        int rowOffset = 0;

        for (Tensor tensor : tensors) {
            int[] shape = tensor.shape(); // [tokens, dimension]

            for (int row = 0; row < shape[0]; row++) {
                for (int col = 0; col < shape[1]; col++) {
                    result.set(tensor.get(row, col), rowOffset + row, col);
                }
            }

            rowOffset += shape[0];
        }

        return result;
    }

    public static Tensor mergeTensors(List<Tensor> tensors) {
        if (tensors.isEmpty()) {
            throw new IllegalArgumentException("No tensors provided!");
        }

        Tensor first = tensors.getFirst();
        int dimension = first.dimension();

        int[] shape = first.shape();
        int[] newShape = new int[dimension + 1];

        newShape[0] = tensors.size();
        System.arraycopy(shape, 0, newShape, 1, dimension);

        Tensor result = zeros(newShape);

        for (int i = 0; i < tensors.size(); i++) {
            Tensor current = tensors.get(i);
            if (current.dimension() != dimension) {
                throw new IllegalArgumentException(
                        "All input tensors must have the same dimension!"
                );
            }

            int[] idx = new int[dimension];
            copyRecursive(current, result, idx, 0, i);
        }

        return result;
    }

    private static void copyRecursive(Tensor src, Tensor dest, int[] idx, int dim, int batchIndex) {
        if (dim == idx.length) {
            double value = src.get(idx);

            int[] destIdx = new int[idx.length + 1];
            destIdx[0] = batchIndex;

            System.arraycopy(idx, 0, destIdx, 1, idx.length);

            dest.set(value, destIdx);
        } else {
            int dimSize = src.shape()[dim];

            for (int j = 0; j < dimSize; j++) {
                idx[dim] = j;
                copyRecursive(src, dest, idx, dim + 1, batchIndex);
            }
        }
    }

    public static Tensor triangularMask(int dimension) {
        Tensor mask = Tensors.zeros(dimension, dimension);

        for (int i = 0; i < dimension; i++) {
            for (int j = i + 1; j < dimension; j++) {
                mask.set(Double.NEGATIVE_INFINITY, i, j);
            }
        }

        return mask;
    }
}