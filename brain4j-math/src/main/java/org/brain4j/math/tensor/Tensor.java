package org.brain4j.math.tensor;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.device.DeviceType;
import org.brain4j.math.lang.DoubleToDoubleFunction;
import org.brain4j.math.tensor.autograd.AutogradContext;
import org.brain4j.math.tensor.autograd.Operation;
import org.brain4j.math.tensor.autograd.operations.*;
import org.brain4j.math.tensor.impl.TensorCPU;
import org.brain4j.math.tensor.impl.TensorGPU;
import org.brain4j.math.tensor.index.Range;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.util.function.BiFunction;
import java.util.function.Supplier;

public interface Tensor extends Iterable<Float> {

    /**
     * Returns the shape of the tensor as an array of integers.
     * @return The shape of the tensor
     */
    int[] shape();

    /**
     * Retrieves the data of the tensor as a float array.
     * @return The tensor's data
     */
    float[] data();

    /**
     * Retrieves the strides of the tensor as an array of integers.
     * @return The strides of the tensor
     */
    int[] strides();

    /**
     * Gets the value at the specified indices in the tensor.
     * @param indices The indices for which the value is requested
     * @return The value at the specified indices
     */
    float get(int... indices);

    /**
     * Returns the number of dimensions of the tensor.
     * @return The number of dimensions
     */
    int dimension();

    /**
     * Returns the number of elements in the tensor.
     * @return The number of elements
     */
    int elements();

    /**
     * Finds the index of the maximum value in the tensor.
     * @return The index of the maximum value
     */
    int argmax();

    /**
     * Sets the value at the specified indices in the tensor.
     * @param value The value to set
     * @param indices The indices where the value should be set
     * @return The current tensor modified
     */
    Tensor set(double value, int... indices);

    /**
     * Adds a value to the tensor at the specified indices.
     * @param value The value to add
     * @param indices The indices where the value should be added
     * @return The current tensor modified
     */
    Tensor add(double value, int... indices);

    /**
     * Creates a clone of the tensor.
     * @return A new tensor that is a clone of this tensor
     */
    Tensor clone();

    /**
     * Adds this tensor with another tensor element-wise.
     * @param other The tensor to add
     * @return The current tensor modified
     */
    Tensor add(Tensor other);

    /**
     * Adds this tensor with a constant value element-wise.
     * @param value The constant value to add
     * @return The current tensor modified
     */
    Tensor add(double value);

    /**
     * Performs element-wise addition of two tensors (alias for `add`).
     * @param other The tensor to add
     * @return A new tensor with the result
     */
    Tensor plus(Tensor other);

    /**
     * Adds a constant value to this tensor element-wise (alias for `add`).
     * @param value The constant value to add
     * @return A new tensor with the result
     */
    Tensor plus(double value);

    /**
     * Subtracts another tensor from this tensor element-wise.
     * @param other The tensor to subtract
     * @return The current tensor modified
     */
    Tensor sub(Tensor other);

    /**
     * Subtracts a constant value from this tensor element-wise.
     * @param value The constant value to subtract
     * @return The current tensor modified
     */
    Tensor sub(double value);

    /**
     * Performs element-wise subtraction of two tensors (alias for `sub`).
     * @param other The tensor to subtract
     * @return A new tensor with the result
     */
    Tensor minus(Tensor other);

    /**
     * Subtracts a constant value from this tensor element-wise (alias for `sub`).
     * @param value The constant value to subtract
     * @return A new tensor with the result
     */
    Tensor minus(double value);

    /**
     * Multiplies this tensor with another tensor element-wise.
     * @param other The tensor to multiply
     * @return The current tensor modified
     */
    Tensor mul(Tensor other);

    /**
     * Multiplies this tensor with a constant value element-wise.
     * @param value The constant value to multiply
     * @return A new tensor with the result
     */
    Tensor mul(double value);

    /**
     * Performs element-wise multiplication of two tensors (alias for `mul`).
     * @param other The tensor to multiply
     * @return A new tensor with the result
     */
    Tensor times(Tensor other);

    /**
     * Multiplies a constant value with this tensor element-wise (alias for `mul`).
     * @param value The constant value to multiply
     * @return A new tensor with the result
     */
    Tensor times(double value);

    /**
     * Divides this tensor by another tensor element-wise.
     * @param other The tensor to divide by
     * @return The current tensor modified
     */
    Tensor div(Tensor other);

    /**
     * Divides this tensor by a constant value element-wise.
     * @param value The constant value to divide by
     * @return The current tensor modified
     */
    Tensor div(double value);

    /**
     * Performs element-wise division of two tensors (alias for `div`).
     * @param other The tensor to divide by
     * @return A new tensor with the result
     */
    Tensor divide(Tensor other);

    /**
     * Divides this tensor by a constant value element-wise (alias for `div`).
     * @param value The constant value to divide by
     * @return A new tensor with the result
     */
    Tensor divide(double value);

    /**
     * Raises each element of the tensor to the power of the given value.
     * @param value The exponent
     * @return A new tensor with the result
     */
    Tensor pow(double value);

    /**
     * Raises each element of the tensor to the power of the corresponding element in another tensor.
     * @param other The tensor containing the exponents
     * @return A new tensor with the result
     */
    Tensor pow(Tensor other);

    /**
     * Takes the square root of each element of the tensor.
     * @return A new tensor with the result
     */
    Tensor sqrt();

    /**
     * Reshapes the tensor to a 1D vector.
     * @return A new tensor
     */
    Tensor vector();

    //=============================================================
    // Linear algebra operations
    //=============================================================

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
    double variance();
    double max();
    double min();
    Tensor sum(int dim, boolean keepDim);
    Tensor mean(int dim, boolean keepDim);
    Tensor sign();

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
    Tensor map(DoubleToDoubleFunction function);
    Tensor fill(float value);
    Tensor fill(Supplier<Double> supplier);

    /**
     * Gets the autograd context for this tensor.
     * @return The autograd context instance
     */
    AutogradContext autogradContext();

    /**
     * Updates the autograd context instance for this tensor.
     * @param autogradContext The new autograd context
     */
    void setAutogradContext(AutogradContext autogradContext);

    /**
     * Enables autograd for this tensor.
     * @return This tensor
     */
    Tensor withGrad();

    /**
     * Returns whether this tensor uses autograd.
     * @return True if this tensor uses autograd, false otherwise
     */
    boolean usesGrad();

    /**
     * Zeros the gradient for this tensor.
     */
    void zerograd();

    /**
     * Gets the gradient for this tensor.
     * @return The gradient
     */
    Tensor grad();

    /**
     * Computes the backward pass for this tensor.
     */
    void backward();

    /**
     * Computes the backward pass for this tensor with the specified gradient.
     * @param gradOutput The gradient
     */
    void backward(Tensor gradOutput);

    /**
     * Executes the specified operation on this tensor and the specified other tensor.
     * @param operation The operation to execute
     * @param other The other tensor
     * @return The result of the operation
     */
    Tensor forward(Operation operation, Tensor other);

    /**
     * Delegates to {@link #forward(Operation, Tensor)} using {@link AddOperation}
     * @return The result of the operation
     */
    Tensor addGrad(Tensor other);

    /**
     * Delegates to {@link #forward(Operation, Tensor)} using {@link MulOperation}
     * @return The result of the operation
     */
    Tensor mulGrad(Tensor other);

    /**
     * Delegates to {@link #forward(Operation, Tensor)} using {@link DivOperation}
     * @param other The other tensor
     * @return The result of the operation
     */
    Tensor divGrad(Tensor other);

    /**
     * Delegates to {@link #forward(Operation, Tensor)} using {@link SubOperation}
     * @param other The other tensor
     * @return The result of the operation
     */
    Tensor subGrad(Tensor other);

    /**
     * Delegates to {@link #forward(Operation, Tensor)} using {@link MatMulOperation}
     * @param other The other tensor
     * @return The result of the operation
     */
    Tensor matmulGrad(Tensor other);

    /**
     * Delegates to {@link #forward(Operation, Tensor)} using {@link ActivationOperation}
     * @param activation The activation to apply
     * @return The resulting tensor
     */
    Tensor activateGrad(Activation activation);

    /**
     * Delegates to {@link #forward(Operation, Tensor)} using {@link TransposeOperation}
     * @return The transposed tensor
     */
    Tensor transposeGrad();

    /**
     * Performs a convolution between this tensor and the specified kernel tensor.
     * Implicitly uses SAME padding and FFT implementation for larger dimensions.
     * Convolution is supported for both 1D and 2D tensors.
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
     * It currently accepts: CPU, GPU, DEFAULT (delegates to CPU)
     * @param deviceType The target device
     * @return The tensor on the target device
     * @throws IllegalArgumentException If the device type is not supported
     */
    default Tensor to(DeviceType deviceType) {
        return switch (deviceType) {
            case CPU, DEFAULT -> TensorCPU.of(shape(), data());
            case GPU -> TensorGPU.fromTensor(this);
            default -> throw new IllegalArgumentException("Unsupported device type: " + deviceType);
        };
    }

    /**
     * Moves the tensor to the GPU if available.
     * @return the tensor on the GPU
     */
    default Tensor gpu() {
        return to(DeviceType.GPU);
    }

    /**
     * Moves the tensor to the CPU.
     * @return the tensor on the CPU
     */
    default Tensor cpu() {
        return to(DeviceType.CPU);
    }

    /**
     * Checks if the tensor is loaded on the specified device type.
     * @param deviceType The device type to check
     * @return True if the tensor is loaded on the specified device, false otherwise
     */
    default boolean isOn(DeviceType deviceType) {
        return switch (deviceType) {
            case CPU -> this instanceof TensorCPU;
            case GPU -> this instanceof TensorGPU;
            default -> throw new IllegalArgumentException("Unsupported device type: " + deviceType);
        };
    }

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
     * Serializes the tensor to the specified label stream.
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