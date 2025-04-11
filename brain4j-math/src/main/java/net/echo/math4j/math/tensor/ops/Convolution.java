package net.echo.math4j.math.tensor.ops;

import net.echo.math4j.math.complex.Complex;
import net.echo.math4j.math.fft.FFT;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;

import static net.echo.math4j.math.constants.Constants.EPSILON;
import static net.echo.math4j.math.constants.Constants.FFT_THRESHOLD;

public final class Convolution {
    
    public enum PaddingMode {
        VALID, 
        SAME,    
        FULL     
    }
    
    public enum ConvolutionType {
        DIRECT,  
        FFT      
    }
    
    private Convolution() {}

    public static Tensor convolve1D(Tensor input, Tensor kernel, 
                                   PaddingMode paddingMode, ConvolutionType convType) {
        if (input.dimension() != 1 || kernel.dimension() != 1) {
            throw new IllegalArgumentException("Input e kernel must be 1D tensors");
        }
        
        int inputSize = input.shape()[0];
        int kernelSize = kernel.shape()[0];
        
        if (convType == null) {
            convType = (kernelSize > FFT_THRESHOLD) ? ConvolutionType.FFT : ConvolutionType.DIRECT;
        }
        
        int outputSize;
        int paddingLeft, paddingRight;
        
        switch (paddingMode) {
            case VALID:
                outputSize = inputSize - kernelSize + 1;
                paddingLeft = 0;
                paddingRight = 0;
                break;
            case SAME:
                outputSize = inputSize;
                int totalPadding = kernelSize - 1;
                paddingLeft = totalPadding / 2;
                paddingRight = totalPadding - paddingLeft;
                break;
            case FULL:
                outputSize = inputSize + kernelSize - 1;
                paddingLeft = kernelSize - 1;
                paddingRight = kernelSize - 1;
                break;
            default:
                throw new IllegalArgumentException("Padding mode not supported");
        }

        if (outputSize <= 0) {
            throw new IllegalArgumentException("Kernel too large for the input with the specified padding");
        }
        
        if (convType == ConvolutionType.DIRECT) {
            return convolve1DDirect(input, kernel, paddingLeft, paddingRight);
        } else {
            return convolve1DFFT(input, kernel, paddingLeft, paddingRight);
        }
    }
    
    private static Tensor convolve1DDirect(Tensor input, Tensor kernel, 
                                          int paddingLeft, int paddingRight) {
        int inputSize = input.shape()[0];
        int kernelSize = kernel.shape()[0];
        int outputSize = inputSize + paddingLeft + paddingRight - kernelSize + 1;
        
        Tensor output = TensorFactory.zeros(outputSize);
        
        float[] flippedKernel = new float[kernelSize];
        for (int i = 0; i < kernelSize; i++) {
            flippedKernel[i] = kernel.get(kernelSize - 1 - i);
        }
        
        for (int i = 0; i < outputSize; i++) {
            double sum = 0;
            for (int j = 0; j < kernelSize; j++) {
                int inputIdx = i - paddingLeft + j;
                if (inputIdx >= 0 && inputIdx < inputSize) {
                    sum += input.get(inputIdx) * flippedKernel[j];
                }
            }
            output.set(sum, i);
        }
        
        return output;
    }
    
    private static Tensor convolve1DFFT(Tensor input, Tensor kernel, 
                                       int paddingLeft, int paddingRight) {
        int inputSize = input.shape()[0];
        int kernelSize = kernel.shape()[0];
        
        int fullSize = inputSize + kernelSize - 1;
        int fftSize = FFT.nextPowerOf2(fullSize);
        
        Complex[] inputComplex = new Complex[fftSize];
        Complex[] kernelComplex = new Complex[fftSize];
        
        for (int i = 0; i < fftSize; i++) {
            inputComplex[i] = Complex.ZERO;
            kernelComplex[i] = Complex.ZERO;
        }
        
        for (int i = 0; i < inputSize; i++) {
            inputComplex[i] = new Complex(input.get(i), 0.0);
        }
        
        for (int i = 0; i < kernelSize; i++) {
            kernelComplex[i] = new Complex(kernel.get(kernelSize - 1 - i), 0.0);
        }
        
        Complex[] inputFFT = FFT.transform(inputComplex);
        Complex[] kernelFFT = FFT.transform(kernelComplex);
        
        Complex[] outputFFT = new Complex[fftSize];
        for (int i = 0; i < fftSize; i++) {
            outputFFT[i] = inputFFT[i].multiply(kernelFFT[i]);
        }
        
        Complex[] result = FFT.inverseTransform(outputFFT);
        
        int startIdx;
        int outputSize;
        
        if (paddingLeft == 0 && paddingRight == 0) {
            startIdx = kernelSize - 1;
            outputSize = inputSize - kernelSize + 1;
        } else if (paddingLeft == (kernelSize - 1) / 2 && paddingRight == (kernelSize - 1) - paddingLeft) {
            startIdx = paddingLeft;
            outputSize = inputSize;
        } else {
            startIdx = 0;
            outputSize = fullSize;
        }
        
        Tensor output = TensorFactory.zeros(outputSize);
        for (int i = 0; i < outputSize; i++) {
            if (startIdx + i < result.length) {
                double val = result[startIdx + i].getReal();
                if (Math.abs(val) < EPSILON) {
                    val = 0.0;
                }
                output.set(val, i);
            }
        }
        
        return output;
    }
    
    public static Tensor convolve2D(Tensor input, Tensor kernel, 
                                   PaddingMode paddingMode, ConvolutionType convType) {
        if (input.dimension() != 2 || kernel.dimension() != 2) {
            throw new IllegalArgumentException("Input and kernel must be 2D tensors");
        }
        
        int[] inputShape = input.shape();
        int[] kernelShape = kernel.shape();
        
        int inputRows = inputShape[0];
        int inputCols = inputShape[1];
        int kernelRows = kernelShape[0];
        int kernelCols = kernelShape[1];
        
        if (convType == null) {
            boolean useFFT = (kernelRows * kernelCols) > FFT_THRESHOLD;
            convType = useFFT ? ConvolutionType.FFT : ConvolutionType.DIRECT;
        }
        
        int outputRows, outputCols;
        int paddingTop, paddingBottom, paddingLeft, paddingRight;
        
        switch (paddingMode) {
            case VALID:
                outputRows = inputRows - kernelRows + 1;
                outputCols = inputCols - kernelCols + 1;
                paddingTop = 0;
                paddingBottom = 0;
                paddingLeft = 0;
                paddingRight = 0;
                break;
            case SAME:
                outputRows = inputRows;
                outputCols = inputCols;
                paddingTop = (kernelRows - 1) / 2;
                paddingBottom = kernelRows - 1 - paddingTop;
                paddingLeft = (kernelCols - 1) / 2;
                paddingRight = kernelCols - 1 - paddingLeft;
                break;
            case FULL:
                outputRows = inputRows + kernelRows - 1;
                outputCols = inputCols + kernelCols - 1;
                paddingTop = kernelRows - 1;
                paddingBottom = kernelRows - 1;
                paddingLeft = kernelCols - 1;
                paddingRight = kernelCols - 1;
                break;
            default:
                throw new IllegalArgumentException("Padding mode not supported");
        }
        
        if (outputRows <= 0 || outputCols <= 0) {
            throw new IllegalArgumentException("Kernel too large for the input with the specified padding");
        }
        
        if (convType == ConvolutionType.DIRECT) {
            return convolve2DDirect(input, kernel, 
                                  paddingTop, paddingBottom, 
                                  paddingLeft, paddingRight);
        } else {
            return convolve2DFFT(input, kernel, 
                               paddingTop, paddingBottom, 
                               paddingLeft, paddingRight);
        }
    }
 
    private static Tensor convolve2DDirect(Tensor input, Tensor kernel,
                                          int paddingTop, int paddingBottom,
                                          int paddingLeft, int paddingRight) {
        int[] inputShape = input.shape();
        int[] kernelShape = kernel.shape();
        
        int inputRows = inputShape[0];
        int inputCols = inputShape[1];
        int kernelRows = kernelShape[0];
        int kernelCols = kernelShape[1];
        
        int outputRows = inputRows + paddingTop + paddingBottom - kernelRows + 1;
        int outputCols = inputCols + paddingLeft + paddingRight - kernelCols + 1;
        
        Tensor output = TensorFactory.zeros(outputRows, outputCols);
        
        float[][] flippedKernel = new float[kernelRows][kernelCols];
        for (int i = 0; i < kernelRows; i++) {
            for (int j = 0; j < kernelCols; j++) {
                flippedKernel[i][j] = kernel.get(kernelRows - 1 - i, kernelCols - 1 - j);
            }
        }
        
        for (int i = 0; i < outputRows; i++) {
            for (int j = 0; j < outputCols; j++) {
                double sum = 0;
                for (int ki = 0; ki < kernelRows; ki++) {
                    for (int kj = 0; kj < kernelCols; kj++) {
                        int ri = i - paddingTop + ki;
                        int cj = j - paddingLeft + kj;
                        
                        if (ri >= 0 && ri < inputRows && cj >= 0 && cj < inputCols) {
                            sum += input.get(ri, cj) * flippedKernel[ki][kj];
                        }
                    }
                }
                output.set(sum, i, j);
            }
        }
        
        return output;
    }
    
    private static Tensor convolve2DFFT(Tensor input, Tensor kernel,
                                       int paddingTop, int paddingBottom,
                                       int paddingLeft, int paddingRight) {
        int[] inputShape = input.shape();
        int[] kernelShape = kernel.shape();
        
        int inputRows = inputShape[0];
        int inputCols = inputShape[1];
        int kernelRows = kernelShape[0];
        int kernelCols = kernelShape[1];
        
        int fullRows = inputRows + kernelRows - 1;
        int fullCols = inputCols + kernelCols - 1;
        
        int fftRows = FFT.nextPowerOf2(fullRows);
        int fftCols = FFT.nextPowerOf2(fullCols);
        
        Complex[][] inputComplex = new Complex[fftRows][fftCols];
        Complex[][] kernelComplex = new Complex[fftRows][fftCols];
        
        for (int i = 0; i < fftRows; i++) {
            for (int j = 0; j < fftCols; j++) {
                inputComplex[i][j] = Complex.ZERO;
                kernelComplex[i][j] = Complex.ZERO;
            }
        }
        
        for (int i = 0; i < inputRows; i++) {
            for (int j = 0; j < inputCols; j++) {
                inputComplex[i][j] = new Complex(input.get(i, j), 0.0);
            }
        }
        
        for (int i = 0; i < kernelRows; i++) {
            for (int j = 0; j < kernelCols; j++) {
                kernelComplex[i][j] = new Complex(
                    kernel.get(kernelRows - 1 - i, kernelCols - 1 - j), 0.0);
            }
        }
        
        Complex[][] inputFFT = FFT.transform2D(inputComplex, fftRows, fftCols);
        Complex[][] kernelFFT = FFT.transform2D(kernelComplex, fftRows, fftCols);
        
        Complex[][] outputFFT = new Complex[fftRows][fftCols];
        for (int i = 0; i < fftRows; i++) {
            for (int j = 0; j < fftCols; j++) {
                outputFFT[i][j] = inputFFT[i][j].multiply(kernelFFT[i][j]);
            }
        }
        
        Complex[][] result = FFT.inverseTransform2D(outputFFT, fftRows, fftCols);
        
        int startRow, startCol;
        int outputRows, outputCols;
        
        if (paddingTop == 0 && paddingBottom == 0 && paddingLeft == 0 && paddingRight == 0) {
            startRow = kernelRows - 1;
            startCol = kernelCols - 1;
            outputRows = inputRows - kernelRows + 1;
            outputCols = inputCols - kernelCols + 1;
        } else if (paddingTop == (kernelRows - 1) / 2 && paddingBottom == (kernelRows - 1) - paddingTop &&
                 paddingLeft == (kernelCols - 1) / 2 && paddingRight == (kernelCols - 1) - paddingLeft) {
            startRow = paddingTop;
            startCol = paddingLeft;
            outputRows = inputRows;
            outputCols = inputCols;
        } else {
            startRow = 0;
            startCol = 0;
            outputRows = fullRows;
            outputCols = fullCols;
        }
        
        Tensor output = TensorFactory.zeros(outputRows, outputCols);
        for (int i = 0; i < outputRows; i++) {
            for (int j = 0; j < outputCols; j++) {
                double val = result[startRow + i][startCol + j].getReal();
                if (Math.abs(val) < EPSILON) {
                    val = 0.0;
                }
                output.set(val, i, j);
            }
        }
        
        return output;
    }
    
    public static Tensor crossCorrelation2D(Tensor input, Tensor kernel, 
                                          PaddingMode paddingMode, ConvolutionType convType) {
        Tensor flippedKernel = flipKernel2D(kernel);
        return convolve2D(input, flippedKernel, paddingMode, convType);
    }
    
    private static Tensor flipKernel2D(Tensor kernel) {
        int[] shape = kernel.shape();
        int rows = shape[0];
        int cols = shape[1];
        
        Tensor flipped = TensorFactory.zeros(rows, cols);
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                flipped.set(kernel.get(rows - 1 - i, cols - 1 - j), i, j);
            }
        }
        
        return flipped;
    }
    
    public static Tensor prepareKernelForFFT(Tensor kernel, int fftRows, int fftCols) {
        int[] kernelShape = kernel.shape();
        int kernelRows = kernelShape[0];
        int kernelCols = kernelShape[1];
        
        Tensor paddedKernel = TensorFactory.zeros(fftRows, fftCols);
        
        for (int i = 0; i < kernelRows; i++) {
            for (int j = 0; j < kernelCols; j++) {
                paddedKernel.set(kernel.get(kernelRows - 1 - i, kernelCols - 1 - j), i, j);
            }
        }
        
        return paddedKernel;
    }
} 