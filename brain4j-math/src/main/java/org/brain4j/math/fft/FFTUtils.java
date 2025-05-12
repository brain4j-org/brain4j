package org.brain4j.math.fft;

import org.brain4j.math.complex.Complex;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

public final class FFTUtils {
    
    private FFTUtils() {
    }
    
    public static Complex[] tensorToComplex1D(Tensor tensor) {
        if (tensor.dimension() != 1) {
            throw new IllegalArgumentException("Input tensor must be 1D");
        }
        
        int size = tensor.shape()[0];
        Complex[] result = new Complex[size];
        
        for (int i = 0; i < size; i++) {
            result[i] = new Complex(tensor.get(i), 0.0);
        }
        
        return result;
    }
    
    public static Tensor complexToTensor1D(Complex[] complex) {
        int size = complex.length;
        Tensor result = Tensors.zeros(size);
        
        for (int i = 0; i < size; i++) {
            result.set(complex[i].getReal(), i);
        }
        
        return result;
    }
    
    public static Complex[][] tensorToComplex2D(Tensor tensor) {
        if (tensor.dimension() != 2) {
            throw new IllegalArgumentException("Input tensor must be 2D");
        }
        
        int[] shape = tensor.shape();
        int rows = shape[0];
        int cols = shape[1];
        
        Complex[][] result = new Complex[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = new Complex(tensor.get(i, j), 0.0);
            }
        }
        
        return result;
    }
    
    public static Tensor complexToTensor2D(
        Complex[][] complex,
        int rows,
        int cols
    ) {
        Tensor result = Tensors.zeros(rows, cols);
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.set(complex[i][j].getReal(), i, j);
            }
        }
        
        return result;
    }
    
    public static Tensor fft1D(Tensor tensor) {
        Complex[] input = tensorToComplex1D(tensor);
        Complex[] output = FFT.transform(input);
        return complexToTensor1D(output);
    }
    
    public static Tensor ifft1D(Tensor tensor) {
        Complex[] input = tensorToComplex1D(tensor);
        Complex[] output = FFT.inverseTransform(input);
        return complexToTensor1D(output);
    }
    
    public static Tensor fft2D(Tensor tensor) {
        Complex[][] input = tensorToComplex2D(tensor);
        int[] shape = tensor.shape();
        Complex[][] output = FFT.transform2D(input, shape[0], shape[1]);
        return complexToTensor2D(output, shape[0], shape[1]);
    }
    
    public static Tensor ifft2D(Tensor tensor) {
        Complex[][] input = tensorToComplex2D(tensor);
        int[] shape = tensor.shape();
        Complex[][] output = FFT.inverseTransform2D(input, shape[0], shape[1]);
        return complexToTensor2D(output, shape[0], shape[1]);
    }
    
    public static Tensor zeroPad1D(Tensor tensor, int size) {
        if (tensor.dimension() != 1) {
            throw new IllegalArgumentException("Input tensor must be 1D");
        }
        
        int currentSize = tensor.shape()[0];
        if (currentSize >= size) {
            return tensor.clone();
        }
        
        Tensor result = Tensors.zeros(size);
        for (int i = 0; i < currentSize; i++) {
            result.set(tensor.get(i), i);
        }
        
        return result;
    }
    
    public static Tensor zeroPad2D(
        Tensor tensor,
        int rows,
        int cols
    ) {
        if (tensor.dimension() != 2) {
            throw new IllegalArgumentException("Input tensor must be 2D");
        }
        
        int[] shape = tensor.shape();
        int currentRows = shape[0];
        int currentCols = shape[1];
        
        if (currentRows >= rows && currentCols >= cols) {
            return tensor.clone();
        }
        
        Tensor result = Tensors.zeros(rows, cols);

        for (int i = 0; i < Math.min(currentRows, rows); i++) {
            for (int j = 0; j < Math.min(currentCols, cols); j++) {
                result.set(tensor.get(i, j), i, j);
            }
        }
        
        return result;
    }
    
    public static Tensor removePadding1D(Tensor tensor, int size) {
        if (tensor.dimension() != 1) {
            throw new IllegalArgumentException("Input tensor must be 1D");
        }
        
        int currentSize = tensor.shape()[0];
        if (currentSize <= size) {
            return tensor.clone();
        }
        
        Tensor result = Tensors.zeros(size);
        for (int i = 0; i < size; i++) {
            result.set(tensor.get(i), i);
        }
        
        return result;
    }
    
    public static Tensor removePadding2D(
        Tensor tensor,
        int rows,
        int cols
    ) {
        if (tensor.dimension() != 2) {
            throw new IllegalArgumentException("Input tensor must be 2D");
        }
        
        int[] shape = tensor.shape();
        int currentRows = shape[0];
        int currentCols = shape[1];
        
        if (currentRows <= rows && currentCols <= cols) {
            return tensor.clone();
        }
        
        Tensor result = Tensors.zeros(Math.min(currentRows, rows), Math.min(currentCols, cols));
        for (int i = 0; i < Math.min(currentRows, rows); i++) {
            for (int j = 0; j < Math.min(currentCols, cols); j++) {
                result.set(tensor.get(i, j), i, j);
            }
        }
        
        return result;
    }
} 