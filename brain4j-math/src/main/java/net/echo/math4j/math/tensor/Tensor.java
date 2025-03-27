package net.echo.math4j.math.tensor;

import net.echo.math4j.math.tensor.autograd.AutogradContext;
import net.echo.math4j.math.tensor.index.Range;
import net.echo.math4j.math.vector.Vector;

import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;

public interface Tensor extends Iterable<Double> {

    //=============================================================
    // Base properties and methods
    //=============================================================
    
    Vector getData();
    int[] shape();
    int dimension();
    int elements();
    float get(int... indices);
    Tensor set(double value, int... indices);
    Tensor add(double value, int... indices);
    float[] toArray();
    double[] toDoubleArray();
    Tensor clone();
    
    //=============================================================
    // Base arithmetic operations
    //=============================================================
    
    // Addition
    Tensor add(Tensor other);
    Tensor add(double value);
    Tensor plus(Tensor other);
    Tensor plus(double value);
    
    // Subtraction
    Tensor sub(Tensor other);
    Tensor sub(double value);
    Tensor minus(Tensor other);
    Tensor minus(double value);
    
    // Multiplication
    Tensor mul(Tensor other);
    Tensor mul(double value);
    Tensor times(Tensor other);
    Tensor times(double value);
    Tensor mul(Vector vec);
    
    // Division
    Tensor div(Tensor other);
    Tensor div(double value);
    Tensor divide(Tensor other);
    Tensor divide(double value);
    
    // Other mathematical operations
    Tensor pow(double value);
    Tensor pow(Tensor other);
    Tensor sqrt();
    
    //=============================================================
    // Linear algebra operations
    //=============================================================
    
    Tensor matmul(Tensor other);
    Tensor matmul(Vector vec);
    double dot(Tensor other);
    double norm();
    double normSquared();
    Tensor normalize();
    double distance(Tensor other);
    double distanceSquared(Tensor other);
    Tensor transpose();
    
    //=============================================================
    // Statistical operations
    //=============================================================
    
    double sum();
    double mean();
    double max();
    double min();
    Tensor sum(int dim, boolean keepDim);
    Tensor mean(int dim, boolean keepDim);
    
    //=============================================================
    // Shape manipulation
    //=============================================================
    
    Tensor reshape(int... newShape);
    Tensor view(int... newShape);
    Tensor permute(int... dims);
    Tensor squeeze();
    Tensor squeeze(int dim);
    Tensor unsqueeze(int dim);
    
    //=============================================================
    // Indexing and selection
    //=============================================================
    
    Tensor select(int dim, int index);
    Tensor slice(int channel);
    Tensor slice(Range... ranges);
    Tensor setChannel(int channel, Tensor data);
    
    //=============================================================
    // Transformation functions
    //=============================================================
    
    Tensor mapWithIndex(BiFunction<Integer, Double, Double> function);
    Tensor map(Function<Double, Double> function);
    Tensor fill(double value);
    Tensor fill(Supplier<Double> supplier);
    
    //=============================================================
    // Autograd operations
    //=============================================================
    
    AutogradContext getAutogradContext();
    void setAutogradContext(AutogradContext autogradContext);
    Tensor requiresGrad(boolean requiresGrad);
    boolean requiresGrad();
    Tensor grad();
    void backward();
    void backward(Tensor gradOutput);
    Tensor addWithGrad(Tensor other);
    Tensor mulWithGrad(Tensor other);
    Tensor divWithGrad(Tensor other);
    Tensor subWithGrad(Tensor other);
    
    //=============================================================
    // Special operations
    //=============================================================
    
    /**
     * Performs a convolution between this tensor and the specified kernel tensor.
     * Implicitly uses SAME padding and FFT implementation for larger dimensions.
     * Convolution is supported for both 1D and 2D tensors.
     *
     * @param kernel the kernel tensor to use for convolution
     * @return a new tensor resulting from the convolution
     * @throws IllegalArgumentException if tensor dimensions are not compatible
     */
    Tensor convolve(Tensor kernel);
    
    Tensor softmax();
    Tensor softmax(double temperature);
    
    //=============================================================
    // Utils and conversion
    //=============================================================
    
    /**
     * Moves the tensor to the GPU if available.
     * @return the tensor on the GPU
     */
    Tensor gpu();

    /**
     * Moves the tensor to the CPU.
     * @return the tensor on the CPU
     */
    Tensor cpu();
    boolean checkNaN();
    String toString(String format);
}