package net.echo.math4j.math.fft;

import net.echo.math4j.math.complex.Complex;
import java.util.Arrays;

public final class FFT {
    private static final double TWO_PI = 2 * Math.PI;
    
    private FFT() {}
    
    public static Complex[] transform(Complex[] input) {
        if (input == null || input.length == 0) {
            throw new IllegalArgumentException("Input array cannot be null or empty");
        }
        
        int n = input.length;
        
        if ((n & (n - 1)) == 0) {
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
        
        if ((n & (n - 1)) != 0) {
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
            Complex twiddle = Complex.fromPolar(1, angle);
            
            result[k] = evenFFT[k].add(twiddle.multiply(oddFFT[k]));
            result[k + n/2] = evenFFT[k].subtract(twiddle.multiply(oddFFT[k]));
        }
        
        return result;
    }
    
    private static Complex[] transformBluestein(Complex[] input) {
        int n = input.length;
        
        int m = 1;
        while (m < 2 * n - 1) {
            m *= 2;
        }
        
        Complex[] a = new Complex[m];
        Arrays.fill(a, Complex.ZERO);
        
        Complex[] b = new Complex[m];
        Arrays.fill(b, Complex.ZERO);
        
        for (int i = 0; i < n; i++) {
            double angle = Math.PI * (i * i) / n;
            Complex chirp = Complex.fromPolar(1, -angle);
            a[i] = input[i].multiply(chirp);
            b[i] = chirp.conjugate();
            b[m - 1 - i] = chirp.conjugate();
        }
        
        Complex[] c = convolveFFT(a, b);
        
        Complex[] result = new Complex[n];
        for (int i = 0; i < n; i++) {
            double angle = Math.PI * (i * i) / n;
            Complex chirp = Complex.fromPolar(1, -angle);
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

    public static boolean isPowerOf2(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }
 
    public static int nextPowerOf2(int n) {
        if (n <= 0) {
            return 1;
        }
        
        n--;
        n |= n >>> 1;
        n |= n >>> 2;
        n |= n >>> 4;
        n |= n >>> 8;
        n |= n >>> 16;
        return n + 1;
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
} 