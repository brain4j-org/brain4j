package net.echo.math4j.math.fft;

import net.echo.math4j.math.complex.Complex;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;

public final class FFTUtils {
    
    private FFTUtils() {}
    
    public static Complex[] tensorToComplex1D(Tensor tensor) {
        if (tensor.dimension() != 1) {
            throw new IllegalArgumentException("Tensor must be 1D");
        }
        
        int size = tensor.shape()[0];
        Complex[] result = new Complex[size];
        
        for (int i = 0; i < size; i++) {
            result[i] = new Complex(tensor.get(i), 0);
        }
        
        return result;
    }
    
    public static Complex[][] tensorToComplex2D(Tensor tensor) {
        if (tensor.dimension() != 2) {
            throw new IllegalArgumentException("Tensor must be 2D");
        }
        
        int[] shape = tensor.shape();
        int rows = shape[0];
        int cols = shape[1];
        
        Complex[][] result = new Complex[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = new Complex(tensor.get(i, j), 0);
            }
        }
        
        return result;
    }
    
    public static Tensor complexToRealTensor1D(Complex[] complexArray) {
        int size = complexArray.length;
        Tensor result = TensorFactory.zeros(size);
        
        for (int i = 0; i < size; i++) {
            result.set(complexArray[i].getReal(), i);
        }
        
        return result;
    }
    
    public static Tensor complexToComplexTensor1D(Complex[] complexArray) {
        int size = complexArray.length;
        Tensor result = TensorFactory.zeros(size, 2);
        
        for (int i = 0; i < size; i++) {
            result.set(complexArray[i].getReal(), i, 0);
            result.set(complexArray[i].getImaginary(), i, 1);
        }
        
        return result;
    }
    
    public static Tensor complexToRealTensor2D(Complex[][] complexArray) {
        int rows = complexArray.length;
        int cols = (rows > 0) ? complexArray[0].length : 0;
        
        Tensor result = TensorFactory.zeros(rows, cols);
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.set(complexArray[i][j].getReal(), i, j);
            }
        }
        
        return result;
    }
    
    public static Tensor complexToComplexTensor2D(Complex[][] complexArray) {
        int rows = complexArray.length;
        int cols = (rows > 0) ? complexArray[0].length : 0;
        
        Tensor result = TensorFactory.zeros(rows, cols, 2);
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.set(complexArray[i][j].getReal(), i, j, 0);
                result.set(complexArray[i][j].getImaginary(), i, j, 1);
            }
        }
        
        return result;
    }
    
    public static Tensor fft1D(Tensor tensor, boolean returnComplex) {
        Complex[] input = tensorToComplex1D(tensor);
        Complex[] output = FFT.transform(input);
        
        return returnComplex ? 
               complexToComplexTensor1D(output) : 
               complexToRealTensor1D(output);
    }
    
    public static Tensor ifft1D(Tensor tensor, boolean isComplex, boolean returnComplex) {
        Complex[] input;
        
        if (isComplex) {
            if (tensor.dimension() != 2 || tensor.shape()[1] != 2) {
                throw new IllegalArgumentException("Tensor must have shape [n, 2]");
            }
            
            int size = tensor.shape()[0];
            input = new Complex[size];
            
            for (int i = 0; i < size; i++) {
                input[i] = new Complex(tensor.get(i, 0), tensor.get(i, 1));
            }
        } else {
            input = tensorToComplex1D(tensor);
        }
        
        Complex[] output = FFT.inverseTransform(input);
        
        return returnComplex ? 
               complexToComplexTensor1D(output) : 
               complexToRealTensor1D(output);
    }
    
    public static Tensor fft2D(Tensor tensor, boolean returnComplex) {
        if (tensor.dimension() != 2) {
            throw new IllegalArgumentException("Tensor must be 2D");
        }
        
        int[] shape = tensor.shape();
        int rows = shape[0];
        int cols = shape[1];
        
        Complex[][] intermediate = new Complex[rows][];
        for (int i = 0; i < rows; i++) {
            Tensor row = tensor.select(0, i);
            Complex[] rowComplex = tensorToComplex1D(row);
            intermediate[i] = FFT.transform(rowComplex);
        }
        
        Complex[][] result = new Complex[rows][cols];
        for (int j = 0; j < cols; j++) {
            Complex[] col = new Complex[rows];
            for (int i = 0; i < rows; i++) {
                col[i] = intermediate[i][j];
            }
            
            Complex[] colTransformed = FFT.transform(col);
            
            for (int i = 0; i < rows; i++) {
                result[i][j] = colTransformed[i];
            }
        }
        
        return returnComplex ? 
               complexToComplexTensor2D(result) : 
               complexToRealTensor2D(result);
    }
    
    public static Tensor ifft2D(Tensor tensor, boolean isComplex, boolean returnComplex) {
        Complex[][] input;
        
        if (isComplex) {
            if (tensor.dimension() != 3 || tensor.shape()[2] != 2) {
                throw new IllegalArgumentException("Tensor must have shape [rows, cols, 2]");
            }
            
            int rows = tensor.shape()[0];
            int cols = tensor.shape()[1];
            input = new Complex[rows][cols];
            
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    input[i][j] = new Complex(tensor.get(i, j, 0), tensor.get(i, j, 1));
                }
            }
        } else {
            input = tensorToComplex2D(tensor);
        }
        
        int rows = input.length;
        int cols = (rows > 0) ? input[0].length : 0;
        
        Complex[][] intermediate = new Complex[rows][cols];
        for (int i = 0; i < rows; i++) {
            Complex[] rowTransformed = FFT.inverseTransform(input[i]);
            intermediate[i] = rowTransformed;
        }
        
        Complex[][] result = new Complex[rows][cols];
        for (int j = 0; j < cols; j++) {
            Complex[] col = new Complex[rows];
            for (int i = 0; i < rows; i++) {
                col[i] = intermediate[i][j];
            }
            
            Complex[] colTransformed = FFT.inverseTransform(col);
            
            for (int i = 0; i < rows; i++) {
                result[i][j] = colTransformed[i];
            }
        }
        
        return returnComplex ? 
               complexToComplexTensor2D(result) : 
               complexToRealTensor2D(result);
    }
} 