package net.echo.math4j.math.tensor;

import net.echo.math4j.math.tensor.autograd.AutogradContext;
import net.echo.math4j.math.tensor.index.Range;
import net.echo.math4j.device.DeviceType;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;

public interface Tensor extends Iterable<Float> {

    //=============================================================
    // Base properties and methods
    //=============================================================
    
    float[] getData();
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

    Tensor oldMatmul(Tensor other);
    Tensor matmul(Tensor other);
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

    Tensor mapWithIndex(BiFunction<Integer, Float, Float> function);
    Tensor map(Function<Double, Double> function);
    Tensor fill(float value);
    Tensor fill(Supplier<Double> supplier);

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

    /**
     * Performs a convolution between this tensor and the specified kernel tensor.
     * Implicitly uses SAME padding and FFT implementation for larger dimensions.
     * Convolution is supported for both 1D and 2D tensors.
     *
     * @param kernel The kernel tensor to use for convolution
     * @return A new tensor resulting from the convolution
     * @throws IllegalArgumentException If tensor dimensions are not compatible
     */
    Tensor convolve(Tensor kernel);

    /**
     * Applies softmax to the tensor with a default temperature of 1.
     * @return The soft-maxed tensor
     */
    Tensor softmax();

    /**
     * Applies softmax to the tensor with the specified temperature.
     * @param temperature A parameter indicating how much to smooth the distribution
     * @return The soft-maxed tensor
     */
    Tensor softmax(double temperature);

    /**
     * Converts a tensor to the specified device type.
     * @param deviceType The target device
     * @return The tensor on the target device
     */
    Tensor to(DeviceType deviceType);

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

    /**
     * Checks if any of the values inside the tensor are NaN.
     * @return True if any of the values are NaN, false otherwise
     */
    boolean checkNaN();

    /**
     * Gets a string containing all the values of this tensor in the specified format.
     * @param format The string format
     * @return The tensor values on a string
     */
    String toString(String format);

    /**
     * Serializes the tensor to the specified output stream.
     * @param stream The stream to write on
     * @throws Exception If serialization fails
     */
    void serialize(DataOutputStream stream) throws Exception;

    /**
     * Deserializes the tensor to the specified input stream.
     * @param stream The stream to read from
     * @return A new tensor with the values from the stream
     * @throws Exception If deserialization fails
     */
    Tensor deserialize(DataInputStream stream) throws Exception;
}