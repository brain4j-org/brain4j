package org.brain4j.math.tensor;

import org.brain4j.math.device.DeviceType;
import org.brain4j.math.tensor.impl.TensorCPU;
import org.brain4j.math.tensor.impl.TensorGPU;
import org.brain4j.math.tensor.index.Range;

import java.util.ArrayList;
import java.util.List;

public class Tensors {
    
    private static boolean useGPU;

    public static void forceCPU() {
        useGPU = false;
    }
    
    public static void useGPUIfAvailable() {
        try {
            useGPU = TensorGPU.isGpuAvailable();
        } catch (Throwable t) {
            System.err.println("GPU acceleration not available: " + t.getMessage());
        }
    }
    
    public static boolean isUsingGPU() {
        return useGPU;
    }

    public static Tensor scalar(double value) {
        return TensorCPU.of(new int[]{1}, value);
    }

    public static Tensor create(int... shape) {
        return useGPU ? new TensorGPU(shape) : new TensorCPU(shape);
    }
    
    public static Tensor of(int[] shape, float... data) {
        return useGPU ? TensorGPU.of(shape, data) : TensorCPU.of(shape, data);
    }
    
    public static Tensor of(int[] shape, double... data) {
        Tensor result = TensorCPU.of(shape, data);
        return useGPU ? TensorGPU.fromTensor(result) : result;
    }
    
    public static Tensor vector(float... data) {
        return of(new int[]{data.length}, data);
    }

    public static Tensor matrix(int rows, int cols, float... data) {
        return of(new int[]{rows, cols}, data);
    }
    
    public static Tensor zeros(int... shape) {
        return useGPU ? new TensorGPU(shape) : new TensorCPU(shape);
    }
    
    public static Tensor ones(int... shape) {
        DeviceType device = useGPU ? DeviceType.GPU : DeviceType.CPU;
        return TensorCPU.ones(shape).to(device);
    }
    
    public static Tensor random(int... shape) {
        Tensor tensor = TensorCPU.random(shape);
        return useGPU ? tensor.gpu() : tensor;
    }

    public static Tensor random(long seed, int... shape) {
        Tensor tensor = TensorCPU.random(seed, shape);
        return useGPU ? tensor.gpu() : tensor;
    }
    
    public static Tensor uniform(double lowerBound, double upperBound, int... shape) {
        Tensor tensor = TensorCPU.uniform(lowerBound, upperBound, shape);
        return useGPU ? tensor.gpu() : tensor;
    }

    public static Tensor random(double mean, double standardDeviation, int... shape) {
        Tensor tensor = TensorCPU.randn(mean, standardDeviation, shape);
        return useGPU ? tensor.gpu() : tensor;
    }

    public static Tensor concat(List<Tensor> inputs) {
        Tensor sample = inputs.getFirst();

        int rows = sample.shape()[0];
        int columns = sample.shape()[1];

        Tensor result = zeros(rows, columns * inputs.size());

        for (int t = 0; t < inputs.size(); t++) {
            Tensor tensor = inputs.get(t);

            if (tensor.shape()[0] != rows || tensor.shape()[1] != columns) {
                throw new IllegalArgumentException("All tensors must have the same shape! (" + rows + "x" + columns + ")");
            }

            int currentColumn = t * columns;

            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    float value = tensor.get(r, c);
                    result.set(value, r, currentColumn + c);
                }
            }
        }

        return result;
    }

    public static List<Tensor> toList(Tensor input) {
        if (input.dimension() == 1) {
            return List.of(input);
        }

        if (input.dimension() != 2) {
            throw new IllegalArgumentException("Tensor must be 1D or 2D!");
        }

        List<Tensor> result = new ArrayList<>();
        
        int rows = input.shape()[0];
        int columns = input.shape()[1];

        for (int i = 0; i < rows; i++) {
            Range range = new Range(i, i + 1);
            Tensor token = input.slice(range);

            token = token.reshape(1, columns);
            result.add(token);
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

        Tensor result = Tensors.create(elements, dimension);
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