package org.brain4j.math.fft;

import org.brain4j.math.complex.Complex;

import java.util.Arrays;

import static org.brain4j.math.Commons.isPowerOf2;
import static org.brain4j.math.Commons.nextPowerOf2;
import static org.brain4j.math.constants.Constants.PI;
import static org.brain4j.math.constants.Constants.TWO_PI;

public final class FFT {
    
    private FFT() {
    }
    
    public static Complex[] transform(Complex[] input) {
        if (input == null || input.length == 0) {
            throw new IllegalArgumentException("Input array cannot be null or empty");
        }
        
        int n = input.length;
        
        if (isPowerOf2(n)) {
            return transformRadix2(input);
        } else {
            return transformBluestein(input);
        }
    }
    
    public static Complex[] inverseTransform(Complex[] input) {
        if (input == null || input.length == 0) {
            throw new IllegalArgumentException("Input array cannot be null or empty");
        }
        
        int n = input.length;
        
        Complex[] conjugatedInput = new Complex[n];
        for (int i = 0; i < n; i++) {
            conjugatedInput[i] = input[i].conjugate();
        }
        
        Complex[] result = transform(conjugatedInput);
        
        for (int i = 0; i < n; i++) {
            result[i] = result[i].conjugate().divide(n);
        }
        
        return result;
    }
  
    private static Complex[] transformRadix2(Complex[] input) {
        int n = input.length;
        
        if (!isPowerOf2(n)) {
            throw new IllegalArgumentException("Array size must be a power of 2");
        }
        
        if (n == 1) {
            return Arrays.copyOf(input, 1);
        }
        
        Complex[] even = new Complex[n/2];
        Complex[] odd = new Complex[n/2];
        for (int i = 0; i < n/2; i++) {
            even[i] = input[2*i];
            odd[i] = input[2*i + 1];
        }
        
        Complex[] evenFFT = transformRadix2(even);
        Complex[] oddFFT = transformRadix2(odd);
        
        Complex[] result = new Complex[n];
        for (int k = 0; k < n/2; k++) {
            double angle = -TWO_PI * k / n;
            Complex twiddle = Complex.fromPolar(1.0, angle);
            
            Complex t = twiddle.multiply(oddFFT[k]);
            result[k] = evenFFT[k].add(t);
            result[k + n/2] = evenFFT[k].subtract(t);
        }
        
        return result;
    }
    
    private static Complex[] transformBluestein(Complex[] input) {
        int n = input.length;
        
        int m = nextPowerOf2(2 * n - 1);
        
        Complex[] a = new Complex[m];
        Arrays.fill(a, Complex.ZERO);
        
        Complex[] b = new Complex[m];
        Arrays.fill(b, Complex.ZERO);
        
        for (int i = 0; i < n; i++) {
            double angle = PI * ((long)i * i % (2 * n)) / n;  // long avoids overflow
            Complex chirp = Complex.fromPolar(1.0, -angle);
            a[i] = input[i].multiply(chirp);
            b[i] = chirp.conjugate();
            
            if (i > 0) {
                b[m - i] = b[i]; // symmetry
            }
        }
        
        Complex[] c = convolveFFT(a, b);
        
        Complex[] result = new Complex[n];
        for (int i = 0; i < n; i++) {
            double angle = PI * ((long)i * i % (2 * n)) / n;
            Complex chirp = Complex.fromPolar(1.0, -angle);
            result[i] = c[i].multiply(chirp);
        }
        
        return result;
    }

    private static Complex[] convolveFFT(Complex[] a, Complex[] b) {
        int n = a.length;
        
        Complex[] aTransform = transformRadix2(a);
        Complex[] bTransform = transformRadix2(b);
        
        Complex[] cTransform = new Complex[n];
        for (int i = 0; i < n; i++) {
            cTransform[i] = aTransform[i].multiply(bTransform[i]);
        }
        
        return inverseTransformRadix2(cTransform);
    }
    
    private static Complex[] inverseTransformRadix2(Complex[] input) {
        int n = input.length;
        
        Complex[] conjugatedInput = new Complex[n];
        for (int i = 0; i < n; i++) {
            conjugatedInput[i] = input[i].conjugate();
        }
        
        Complex[] result = transformRadix2(conjugatedInput);
        
        for (int i = 0; i < n; i++) {
            result[i] = result[i].conjugate().divide(n);
        }
        
        return result;
    }
    
    public static Complex[] zeroPad(Complex[] input, int size) {
        if (input == null) {
            throw new IllegalArgumentException("Input array cannot be null");
        }
        
        if (input.length >= size) {
            return Arrays.copyOf(input, size);
        }
        
        Complex[] result = new Complex[size];
        System.arraycopy(input, 0, result, 0, input.length);
        
        for (int i = input.length; i < size; i++) {
            result[i] = Complex.ZERO;
        }
        
        return result;
    }
    
    public static Complex[][] transform2D(
        Complex[][] input,
        int rows,
        int cols
    ) {
        if (input == null || rows <= 0 || cols <= 0) {
            throw new IllegalArgumentException("Invalid input for 2D FFT");
        }
        
        Complex[][] temp = new Complex[rows][cols];
        for (int i = 0; i < rows; i++) {
            temp[i] = transform(input[i]);
        }
        
        Complex[][] result = new Complex[rows][cols];
        for (int j = 0; j < cols; j++) {
            Complex[] column = new Complex[rows];
            for (int i = 0; i < rows; i++) {
                column[i] = temp[i][j];
            }
            
            Complex[] transformedColumn = transform(column);
            
            for (int i = 0; i < rows; i++) {
                result[i][j] = transformedColumn[i];
            }
        }
        
        return result;
    }
    
    public static Complex[][] inverseTransform2D(
        Complex[][] input,
        int rows,
        int cols
    ) {
        if (input == null || rows <= 0 || cols <= 0) {
            throw new IllegalArgumentException("Invalid input for 2D inverse FFT");
        }
        
        Complex[][] temp = new Complex[rows][cols];
        for (int i = 0; i < rows; i++) {
            temp[i] = inverseTransform(input[i]);
        }
        
        Complex[][] result = new Complex[rows][cols];

        for (int j = 0; j < cols; j++) {
            Complex[] column = new Complex[rows];

            for (int i = 0; i < rows; i++) {
                column[i] = temp[i][j];
            }
            
            Complex[] transformedColumn = inverseTransform(column);
            
            for (int i = 0; i < rows; i++) {
                result[i][j] = transformedColumn[i];
            }
        }
        
        return result;
    }
} 