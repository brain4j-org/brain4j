package org.brain4j.math.tensor;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.lang.DoubleToDoubleFunction;
import org.brain4j.math.tensor.autograd.AutogradContext;
import org.brain4j.math.tensor.autograd.Operation;
import org.brain4j.math.tensor.autograd.operations.*;
import org.brain4j.math.tensor.index.Range;

import java.util.Arrays;
import java.util.function.Supplier;

public interface Tensor extends Iterable<Float> {

    /**
     * Returns the shape of the tensor as an array of integers.
     * @return the shape of the tensor
     */
    int[] shape();

    /**
     * Retrieves the data of the tensor as a float array.
     * @return the tensor's data
     */
    float[] data();

    /**
     * Retrieves the strides of the tensor as an array of integers.
     * @return the strides of the tensor
     */
    int[] strides();

    /**
     * Gets the linear index of the specified indices in the tensor.
     * @param indices the multidimensional indices of the tensor
     * @return a linear index that represent the specified indices in 1 dimension
     */
    int getLinearIndex(int... indices);

    /**
     * Gets the value at the specified indices in the tensor.
     * @param indices The indices for which the value is requested.
     * @return The value at the specified indices.
     */
    float get(int... indices);

    /**
     * Returns the number of dimensions of the tensor.
     * @return The number of dimensions.
     */
    int dimension();

    /**
     * Returns the number of elements in the tensor.
     * @return The number of elements.
     */
    int elements();

    /**
     * Finds the index of the maximum value in the tensor.
     * @return The index of the maximum value.
     */
    int argmax();

    /**
     * Sets the value at the specified indices in the tensor.
     * @param value The value to set.
     * @param indices The indices where the value should be set.
     * @return The current tensor modified.
     */
    Tensor set(double value, int... indices);

    /**
     * Adds a value to the tensor at the specified indices.
     * @param value The value to add.
     * @param indices The indices where the value should be added.
     * @return The current tensor modified.
     */
    Tensor add(double value, int... indices);

    /**
     * Creates a clone of the tensor.
     * @return A new tensor that is a clone of this tensor.
     */
    Tensor clone();

    /**
     * Adds this tensor with another tensor element-wise.
     * @param other The tensor to add.
     * @return The current tensor modified.
     */
    Tensor add(Tensor other);

    /**
     * Adds this tensor with a constant value element-wise.
     * @param value The constant value to add.
     * @return The current tensor modified.
     */
    Tensor add(double value);

    /**
     * Performs element-wise addition of two tensors (alias for `add`).
     * @param other The tensor to add.
     * @return A new tensor with the result.
     */
    default Tensor plus(Tensor other) {
        return clone().add(other);
    }

    /**
     * Adds a constant value to this tensor element-wise (alias for `add`).
     * @param value The constant value to add.
     * @return A new tensor with the result.
     */
    default Tensor plus(double value) {
        return clone().add(value);
    }

    /**
     * Subtracts another tensor from this tensor element-wise.
     * @param other The tensor to subtract.
     * @return The current tensor modified.
     */
    Tensor sub(Tensor other);

    /**
     * Subtracts a constant value from this tensor element-wise.
     * @param value The constant value to subtract.
     * @return The current tensor modified.
     */
    Tensor sub(double value);

    /**
     * Performs element-wise subtraction of two tensors (alias for `sub`).
     * @param other The tensor to subtract.
     * @return A new tensor with the result.
     */
    default Tensor minus(Tensor other) {
        return clone().sub(other);
    }

    /**
     * Subtracts a constant value from this tensor element-wise (alias for `sub`).
     * @param value The constant value to subtract.
     * @return A new tensor with the result.
     */
    default Tensor minus(double value) {
        return clone().sub(value);
    }

    /**
     * Multiplies this tensor with another tensor element-wise.
     * @param other The tensor to multiply.
     * @return The current tensor modified.
     */
    Tensor mul(Tensor other);

    /**
     * Multiplies this tensor with a constant value element-wise.
     * @param value The constant value to multiply.
     * @return A new tensor with the result.
     */
    Tensor mul(double value);

    /**
     * Performs element-wise multiplication of two tensors (alias for `mul`).
     * @param other The tensor to multiply.
     * @return A new tensor with the result.
     */
    default Tensor times(Tensor other) {
        return clone().mul(other);
    }

    /**
     * Multiplies a constant value with this tensor element-wise (alias for `mul`).
     * @param value The constant value to multiply.
     * @return A new tensor with the result.
     */
    default Tensor times(double value) {
        return clone().mul(value);
    }

    /**
     * Divides this tensor by another tensor element-wise.
     * @param other The tensor to divide by.
     * @return The current tensor modified.
     */
    Tensor div(Tensor other);

    /**
     * Divides this tensor by a constant value element-wise.
     * @param value The constant value to divide by.
     * @return The current tensor modified.
     */
    Tensor div(double value);

    /**
     * Performs element-wise division of two tensors (alias for `div`).
     * @param other The tensor to divide by.
     * @return A new tensor with the result.
     */
    default Tensor divide(Tensor other) {
        return clone().div(other);
    }

    /**
     * Divides this tensor by a constant value element-wise (alias for `div`).
     * @param value The constant value to divide by.
     * @return A new tensor with the result.
     */
    default Tensor divide(double value) {
        return clone().div(value);
    }

    /**
     * Raises each element of the tensor to the power of the given value.
     * @param value The exponent.
     * @return A new tensor with the result.
     */
    Tensor pow(double value);

    /**
     * Raises each element of the tensor to the power of the corresponding element in another tensor.
     * @param other The tensor containing the exponents.
     * @return A new tensor with the result.
     */
    Tensor pow(Tensor other);

    /**
     * Takes the square root of each element of the tensor.
     * @return A new tensor with the result.
     */
    Tensor sqrt();

    /**
     * Reshapes the tensor to a 1D vector.
     * @return A new tensor.
     */
    Tensor vector();

    //=============================================================
    // Linear algebra operations
    //=============================================================

    Tensor matmul(Tensor other);
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

    /**
     * Activates all the elements of this tensor using the specified activation function.
     * @param activation the activation function
     * @return the resulting tensor
     */
    Tensor activate(Activation activation);

    /**
     * Selects a sub-tensor from this tensor, given the specified dimension
     * and index.
     *
     * @param dim The dimension to select from.
     * @param index The index in the specified dimension to select.
     * @return A new tensor with the selected values.
     */
    Tensor select(int dim, int index);

    /**
     * Slices the tensor according to the specified ranges for each dimension.
     *
     * @param ranges The ranges specifying the slice for each dimension.
     * @return A new tensor containing the sliced data.
     * @throws IllegalArgumentException if more ranges are specified than the number of dimensions.
     */
    Tensor slice(Range... ranges);

    /**
     * Slices the tensor along the last dimension, starting at the specified offset.
     *
     * @param offset the offset along the last dimension
     * @param input the tensor to slice
     */
    void setSliceAlongLastDim(int offset, Tensor input);

    /**
     * Applies a given function to each element of the tensor and returns a new tensor with the results.
     * @param function The function to apply.
     * @return A new tensor with the result.
     */
    Tensor map(DoubleToDoubleFunction function);

    /**
     * Sets all elements of this tensor to the given value.
     * @param value The value to fill the tensor with.
     * @return This tensor.
     */
    Tensor fill(float value);

    /**
     * Fills the tensor with the values generated by the given supplier.
     * @param supplier The supplier of values.
     * @return The filled tensor.
     */
    Tensor fill(Supplier<Double> supplier);

    /**
     * Gets the autograd context for this tensor.
     * @return The autograd context instance.
     */
    AutogradContext autogradContext();

    /**
     * Updates the autograd context instance for this tensor.
     * @param autogradContext The new autograd context.
     */
    void setAutogradContext(AutogradContext autogradContext);

    /**
     * Enables autograd for this tensor.
     * @return This tensor.
     */
    Tensor withGrad();

    /**
     * Returns whether this tensor uses autograd.
     * @return True if this tensor uses autograd, false otherwise.
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
     * @param gradOutput The gradient.
     */
    void backward(Tensor gradOutput);

    /**
     * Executes the specified operation on this tensor and the specified other tensor.
     * @param operation the operation to execute.
     * @param other the other tensor.
     * @return the result of the operation.
     */
    Tensor forward(Operation operation, Tensor other);

    /**
     * Executes the specified operation on this tensor and the specified other tensors.
     * @param operation the operation to execute.
     * @param others the other tensors.
     * @return the result of the operation.
     */
    Tensor forward(Operation operation, Tensor[] others);

    /**
     * Delegates to {@link #forward(Operation, Tensor)} using {@link AddOperation}
     * @return The result of the operation.
     */
    Tensor addGrad(Tensor other);

    /**
     * Delegates to {@link #forward(Operation, Tensor)} using {@link MulOperation}
     * @return The result of the operation.
     */
    Tensor mulGrad(Tensor other);

    /**
     * Delegates to {@link #forward(Operation, Tensor)} using {@link DivOperation}
     * @param other The other tensor.
     * @return The result of the operation.
     */
    Tensor divGrad(Tensor other);

    /**
     * Delegates to {@link #forward(Operation, Tensor)} using {@link SubOperation}
     * @param other The other tensor.
     * @return The result of the operation.
     */
    Tensor subGrad(Tensor other);

    /**
     * Delegates to {@link #forward(Operation, Tensor)} using {@link MatMulOperation}
     * @param other The other tensor.
     * @return The result of the operation.
     */
    Tensor matmulGrad(Tensor other);

    /**
     * Delegates to {@link #forward(Operation, Tensor)} using {@link ActivationOperation}
     * @param activation The activation to apply.
     * @return The resulting tensor.
     */
    Tensor activateGrad(Activation activation);

    /**
     * Performs a convolution between this tensor and the specified kernel tensor.
     * Implicitly uses SAME padding and FFT implementation for larger dimensions.
     * Convolution is supported for both 1D and 2D tensors.
     *
     * @param kernel the kernel tensor to use for convolution.
     * @param stride the stride to use for the convolution
     * @param padding the padding to use for the convolution
     * @return a new tensor resulting from the convolution.
     * @throws IllegalArgumentException if tensor dimensions are not compatible.
     */
    Tensor convolve(Tensor kernel, int stride, int padding);

    /**
     * Applies softmax to the tensor with a default temperature of 1.
     * @return The soft-maxed tensor.
     */
    Tensor softmax();

    /**
     * Applies softmax to the tensor with the specified temperature.
     * @param temperature A parameter indicating how much to smooth the distribution.
     * @return The soft-maxed tensor.
     */
    Tensor softmax(double temperature);

    /**
     * Gets a string containing all the values of this tensor in the specified format.
     * @param format The string format.
     * @return The tensor values on a string.
     */
    String toString(String format);
}