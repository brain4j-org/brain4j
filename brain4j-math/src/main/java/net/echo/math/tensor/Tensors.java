package net.echo.math.tensor;

import net.echo.math.device.DeviceType;
import net.echo.math.tensor.impl.TensorCPU;
import net.echo.math.tensor.impl.TensorGPU;
import net.echo.math.tensor.index.Range;

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
        return useGPU ? result.gpu() : result;
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

    public static Tensor mergeTensors(List<Tensor> tensors) {
        int rows = tensors.size();
        int columns = tensors.getFirst().shape()[1];

        Tensor result = zeros(rows, columns);

        for (int i = 0; i < rows; i++) {
            Tensor token = tensors.get(i);

            for (int j = 0; j < columns; j++) {
                result.set(token.get(0, j), i, j);
            }
        }

        return result;
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