package org.brain4j.math.complex;

import java.io.Serializable;
import java.util.Objects;

public final class Complex implements Serializable {
    
    public static final Complex ZERO = new Complex(0, 0);
    public static final Complex ONE = new Complex(1, 0);
    public static final Complex I = new Complex(0, 1);
    
    private final double real;      
    private final double imaginary; 
    
    public Complex(double real, double imaginary) {
        this.real = real;
        this.imaginary = imaginary;
    }
    
    public Complex(double real) {
        this(real, 0);
    }
    
    public double getReal() {
        return real;
    }
    
    public double getImaginary() {
        return imaginary;
    }

    public double abs() {
        return Math.hypot(real, imaginary);
    }
    
    public double absSquared() {
        return (real * real) + (imaginary * imaginary);
    }
    
    public double arg() {
        return Math.atan2(imaginary, real);
    }
    
    public Complex conjugate() {
        return new Complex(real, -imaginary);
    }
    
    public Complex add(Complex c) {
        Objects.requireNonNull(c, "Parameter c cannot be null");
        return new Complex(real + c.real, imaginary + c.imaginary);
    }
    
    public Complex add(double real) {
        return new Complex(this.real + real, this.imaginary);
    }
    
    public Complex subtract(Complex c) {
        Objects.requireNonNull(c, "Parameter c cannot be null");
        return new Complex(real - c.real, imaginary - c.imaginary);
    }
    
    public Complex subtract(double real) {
        return new Complex(this.real - real, this.imaginary);
    }
    
    public Complex multiply(Complex c) {
        Objects.requireNonNull(c, "Parameter c cannot be null");
        double newReal = this.real * c.real - this.imaginary * c.imaginary;
        double newImaginary = this.real * c.imaginary + this.imaginary * c.real;
        return new Complex(newReal, newImaginary);
    }
    
    public Complex multiply(double scalar) {
        return new Complex(real * scalar, imaginary * scalar);
    }
    
    public Complex divide(Complex c) {
        Objects.requireNonNull(c, "Parameter c cannot be null");
        
        double denominator = c.absSquared();
        if (denominator == 0) {
            throw new ArithmeticException("Division by zero");
        }
        
        double newReal = (real * c.real + imaginary * c.imaginary) / denominator;
        double newImaginary = (imaginary * c.real - real * c.imaginary) / denominator;
        
        return new Complex(newReal, newImaginary);
    }
    
    public Complex divide(double scalar) {
        if (scalar == 0) {
            throw new ArithmeticException("Division by zero");
        }
        return new Complex(real / scalar, imaginary / scalar);
    }
    
    public Complex pow(int n) {
        if (n == 0) {
            return ONE;
        }
        
        if (n < 0) {
            return ONE.divide(this.pow(-n));
        }
        
        double r = abs();
        double theta = arg();
        
        double newR = Math.pow(r, n);
        double newTheta = n * theta;
        
        return fromPolar(newR, newTheta);
    }
    
    public Complex exp() {
        double expReal = Math.exp(real);
        return new Complex(expReal * Math.cos(imaginary), expReal * Math.sin(imaginary));
    }
    
    public Complex log() {
        if (real == 0 && imaginary == 0) {
            throw new ArithmeticException("Logarithm of zero");
        }
        return new Complex(Math.log(abs()), arg());
    }

    public Complex sqrt() {
        double r = abs();
        
        if (r == 0) {
            return ZERO;
        }
        
        double theta = arg() / 2;
        double sqrtR = Math.sqrt(r);
        
        return new Complex(sqrtR * Math.cos(theta), sqrtR * Math.sin(theta));
    }
    
    public Complex sin() {
        return new Complex(Math.sin(real) * Math.cosh(imaginary), 
                          Math.cos(real) * Math.sinh(imaginary));
    }
    
    public Complex cos() {
        return new Complex(Math.cos(real) * Math.cosh(imaginary),
                         -Math.sin(real) * Math.sinh(imaginary));
    }
    
    public Complex tan() {
        return sin().divide(cos());
    }
    
    public static Complex fromPolar(double r, double theta) {
        return new Complex(r * Math.cos(theta), r * Math.sin(theta));
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        
        Complex complex = (Complex) o;
        
        return Double.compare(complex.real, real) == 0 &&
               Double.compare(complex.imaginary, imaginary) == 0;
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(real, imaginary);
    }
    
    @Override
    public String toString() {
        if (imaginary == 0) {
            return Double.toString(real);
        }
        
        if (real == 0) {
            return imaginary + "i";
        }
        
        if (imaginary < 0) {
            return real + " - " + (-imaginary) + "i";
        }
        
        return real + " + " + imaginary + "i";
    }
} 