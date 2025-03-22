package net.echo.math4j.math.tensor.ops;

import net.echo.math4j.math.complex.Complex;
import net.echo.math4j.math.fft.FFT;
import net.echo.math4j.math.fft.FFTUtils;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;

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
        
        int outputSize;
        switch (paddingMode) {
            case VALID:
                outputSize = inputSize - kernelSize + 1;
                break;
            case SAME:
                outputSize = inputSize;
                break;
            case FULL:
                outputSize = inputSize + kernelSize - 1;
                break;
            default:
                throw new IllegalArgumentException("Padding mode not supported");
        }
        
        if (outputSize <= 0) {
            throw new IllegalArgumentException("Kernel too large for the input with the specified padding");
        }
        
        int paddingLeft, paddingRight;
        switch (paddingMode) {
            case VALID:
                paddingLeft = 0;
                paddingRight = 0;
                break;
            case SAME:
                int totalPadding = kernelSize - 1;
                paddingLeft = totalPadding / 2;
                paddingRight = totalPadding - paddingLeft;
                break;
            case FULL:
                paddingLeft = kernelSize - 1;
                paddingRight = kernelSize - 1;
                break;
            default:
                throw new IllegalArgumentException("Padding mode not supported");
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
        
        for (int i = 0; i < outputSize; i++) {
            double sum = 0;
            for (int j = 0; j < kernelSize; j++) {
                int inputIdx = i - paddingLeft + j;
                if (inputIdx >= 0 && inputIdx < inputSize) {
                    sum += input.get(inputIdx) * kernel.get(j);
                }
            }
            output.set(sum, i);
        }
        
        return output;
    }
    
    private static Tensor normalizeFFTResult(Tensor tensorFFT) {
        float maxAbs = 0.0f;
        for (int i = 0; i < tensorFFT.elements(); i++) {
            float val = Math.abs(tensorFFT.get(i));
            if (val > maxAbs) {
                maxAbs = val;
            }
        }
        
        if (maxAbs < 1e-6f) {
            return tensorFFT;
        }
        
        Tensor result = TensorFactory.zeros(tensorFFT.shape());
        for (int i = 0; i < result.elements(); i++) {
            float normalizedVal = tensorFFT.get(i);
            if (Math.abs(normalizedVal) < 1e-6f) {
                normalizedVal = 0.0f;
            }
            result.set(normalizedVal, i);
        }
        
        return result;
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
            inputComplex[i] = new Complex(0, 0);
            kernelComplex[i] = new Complex(0, 0);
        }
        
        for (int i = 0; i < inputSize; i++) {
            inputComplex[i] = new Complex(input.get(i), 0);
        }
        
        for (int i = 0; i < kernelSize; i++) {
            kernelComplex[i] = new Complex(kernel.get(i), 0);
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
                output.set(result[startIdx + i].getReal(), i);
            }
        }
        
        return normalizeFFTResult(output);
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
        
        int outputRows, outputCols;
        switch (paddingMode) {
            case VALID:
                outputRows = inputRows - kernelRows + 1;
                outputCols = inputCols - kernelCols + 1;
                break;
            case SAME:
                outputRows = inputRows;
                outputCols = inputCols;
                break;
            case FULL:
                outputRows = inputRows + kernelRows - 1;
                outputCols = inputCols + kernelCols - 1;
                break;
            default:
                throw new IllegalArgumentException("Padding mode not supported");
        }
        
        if (outputRows <= 0 || outputCols <= 0) {
            throw new IllegalArgumentException("Kernel too large for the input with the specified padding");
        }
        
        int paddingTop, paddingBottom, paddingLeft, paddingRight;
        switch (paddingMode) {
            case VALID:
                paddingTop = 0;
                paddingBottom = 0;
                paddingLeft = 0;
                paddingRight = 0;
                break;
            case SAME:
                paddingTop = (kernelRows - 1) / 2;
                paddingBottom = kernelRows - 1 - paddingTop;
                paddingLeft = (kernelCols - 1) / 2;
                paddingRight = kernelCols - 1 - paddingLeft;
                break;
            case FULL:
                paddingTop = kernelRows - 1;
                paddingBottom = kernelRows - 1;
                paddingLeft = kernelCols - 1;
                paddingRight = kernelCols - 1;
                break;
            default:
                throw new IllegalArgumentException("Padding mode not supported");
        }
        
        if (convType == ConvolutionType.DIRECT) {
            return convolve2DDirect(input, kernel, paddingTop, paddingBottom, paddingLeft, paddingRight);
        } else {
            return convolve2DFFT(input, kernel, paddingTop, paddingBottom, paddingLeft, paddingRight);
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
        
        int paddedRows = inputRows + paddingTop + paddingBottom;
        int paddedCols = inputCols + paddingLeft + paddingRight;
        
        int outputRows = paddedRows - kernelRows + 1;
        int outputCols = paddedCols - kernelCols + 1;
        
        Tensor output = TensorFactory.zeros(outputRows, outputCols);
        
        for (int i = 0; i < outputRows; i++) {
            for (int j = 0; j < outputCols; j++) {
                double sum = 0;
                for (int ki = 0; ki < kernelRows; ki++) {
                    for (int kj = 0; kj < kernelCols; kj++) {
                        int inputRowIdx = i - paddingTop + ki;
                        int inputColIdx = j - paddingLeft + kj;
                        
                        if (inputRowIdx >= 0 && inputRowIdx < inputRows &&
                            inputColIdx >= 0 && inputColIdx < inputCols) {
                            sum += input.get(inputRowIdx, inputColIdx) * 
                                  kernel.get(ki, kj);
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
        
        int paddedRows = inputRows + paddingTop + paddingBottom;
        int paddedCols = inputCols + paddingLeft + paddingRight;
        
        int fftRows = FFT.nextPowerOf2(paddedRows + kernelRows - 1);
        int fftCols = FFT.nextPowerOf2(paddedCols + kernelCols - 1);
        
        Tensor paddedInput = TensorFactory.zeros(paddedRows, paddedCols);
        for (int i = 0; i < inputRows; i++) {
            for (int j = 0; j < inputCols; j++) {
                paddedInput.set(input.get(i, j), i + paddingTop, j + paddingLeft);
            }
        }
        
        Tensor fftSizeInput = TensorFactory.zeros(fftRows, fftCols);
        Tensor fftSizeKernel = TensorFactory.zeros(fftRows, fftCols);
        
        for (int i = 0; i < paddedRows; i++) {
            for (int j = 0; j < paddedCols; j++) {
                fftSizeInput.set(paddedInput.get(i, j), i, j);
            }
        }
        
        for (int i = 0; i < kernelRows; i++) {
            for (int j = 0; j < kernelCols; j++) {
                fftSizeKernel.set(kernel.get(i, j), i, j);
            }
        }
        
        Tensor inputFFT = FFTUtils.fft2D(fftSizeInput, true);
        Tensor kernelFFT = FFTUtils.fft2D(fftSizeKernel, true);
        
        Tensor outputFFT = TensorFactory.zeros(fftRows, fftCols, 2);
        for (int i = 0; i < fftRows; i++) {
            for (int j = 0; j < fftCols; j++) {
                Complex a = new Complex(inputFFT.get(i, j, 0), inputFFT.get(i, j, 1));
                Complex b = new Complex(kernelFFT.get(i, j, 0), kernelFFT.get(i, j, 1));
                Complex c = a.multiply(b);
                
                outputFFT.set(c.getReal(), i, j, 0);
                outputFFT.set(c.getImaginary(), i, j, 1);
            }
        }
        
        Tensor result = FFTUtils.ifft2D(outputFFT, true, false);
        
        int outputRows = paddedRows - kernelRows + 1;
        int outputCols = paddedCols - kernelCols + 1;
        
        Tensor output = TensorFactory.zeros(outputRows, outputCols);
        for (int i = 0; i < outputRows; i++) {
            for (int j = 0; j < outputCols; j++) {
                output.set(result.get(i, j), i, j);
            }
        }
        
        return output;
    }
    
    public static Tensor crossCorrelation2D(Tensor input, Tensor kernel, 
                                           PaddingMode paddingMode, ConvolutionType convType) {
        return convolve2D(input, kernel, paddingMode, convType);
    }
    
    public static Tensor prepareKernelForFFT(Tensor kernel, int fftRows, int fftCols) {
        int[] kernelShape = kernel.shape();
        int kernelRows = kernelShape[0];
        int kernelCols = kernelShape[1];
        
        Tensor fftSizeKernel = TensorFactory.zeros(fftRows, fftCols);
        
        for (int i = 0; i < kernelRows; i++) {
            for (int j = 0; j < kernelCols; j++) {
                fftSizeKernel.set(kernel.get(i, j), i, j);
            }
        }
        
        return FFTUtils.fft2D(fftSizeKernel, true);
    }
} 