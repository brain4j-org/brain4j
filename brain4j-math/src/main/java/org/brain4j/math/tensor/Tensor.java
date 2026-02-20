package org.brain4j.math.tensor;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.impl.ReLU;
import org.brain4j.math.activation.impl.Sigmoid;
import org.brain4j.math.activation.impl.Tanh;
import org.brain4j.math.commons.D2DFunction;
import org.brain4j.math.tensor.autograd.AutogradContext;
import org.brain4j.math.tensor.autograd.Operation;
import org.brain4j.math.tensor.autograd.impl.*;
import org.brain4j.math.commons.Range;

import java.util.function.Supplier;

/**
 * A multidimensional array that supports vectorized operations.
 *
 * <p>The Tensor interface provides:
 * <ul>
 *   <li>Vectorized arithmetic operations (add, multiply, divide etc)
 *   <li>Matrix operations (matmul, transpose, etc)
 *   <li>Shape manipulation (reshape, squeeze, etc)
 *   <li>Automatic differentiation support
 *   <li>CPU and GPU backend compatibility
 * </ul>
 *
 * <p>Tensors can be created via factory methods in {@link org.brain4j.math.Tensors}:
 * <pre>{@code
 * // Create tensors
 * Tensor vector = Tensors.vector(1, 2, 3, 4);
 * Tensor matrix = Tensors.matrix(2, 2, new float[]{1,2,3,4});
 *
 * // Perform operations
 * Tensor result = matrix.matmul(vector);
 * }</pre>
 *
 * <p>The framework automatically handles moving tensors between CPU and GPU
 * as needed based on the operations being performed.
 *
 * @apiNote unless specified otherwise, all operations which return
 *          a new tensor do not retain the autograd context from the input
 */
public interface Tensor extends Iterable<Float> {

    /**
     * Returns the size of the specified dimension.
     *
     * @param index dimension index
     * @return size of the dimension
     * @throws IndexOutOfBoundsException if index >= rank()
     */
    int shapeAt(int index);

    /**
     * Returns the shape (dimensions) of this tensor.
     * <p>
     * For example, a 2x3 matrix would return [2, 3], while a
     * 3D tensor of size 2x3x4 would return [2, 3, 4].
     *
     * @return array containing size of each dimension
     */
    int[] shape();

    /**
     * Returns the raw data buffer of this tensor.
     * <p>
     * The returned array contains the tensor elements in row-major order.
     * Note that for efficiency this may return the internal array - do not modify
     * the returned array unless you are certain that is safe.
     *
     * @return float array containing tensor data
     */
    float[] data();

    /**
     * Returns the stride (step size) for each dimension.
     * <p>
     * The stride array indicates how many elements to skip to move
     * to the next position in each dimension. For example, in a
     * 2x3 matrix stored in row-major order, the strides would be [3,1].
     *
     * @return array containing stride for each dimension
     */
    int[] strides();
    
    /**
     * Converts the tensor data to a byte array.
     * @return a byte array
     */
    byte[] toByteArray();

    /**
     * Gets the linear index of the specified indices in the tensor.
     * @param indices the multidimensional indices of the tensor
     * @return a linear index that represent the specified indices in 1 dimension
     */
    int linearIndex(int... indices);

    /**
     * Gets the value at the specified indices in the tensor.
     * @param indices the indices for which the value is requested
     * @return the value at the specified indices
     */
    float get(int... indices);

    /**
     * Sets the value at the specified indices in the tensor.
     * @param value the value to set
     * @param indices the indices where the value should be set
     * @return the current tensor modified
     */
    Tensor set(float value, int... indices);
    
    /**
     * Sets the value at the specified indices in the tensor.
     * @param value the value to set
     * @param indices the indices where the value should be set
     * @return the current tensor modified
     */
    default Tensor set(double value, int... indices) {
        return set((float) value, indices);
    }

    /**
     * Returns the number of dimensions (rank) of this tensor.
     * <p>
     * For example:
     * <ul>
     *   <li>A scalar has rank 0
     *   <li>A vector has rank 1
     *   <li>A matrix has rank 2
     *   <li>A 3D tensor has rank 3
     * </ul>
     *
     * @return number of dimensions
     */
    int rank();

    /**
     * Returns the total number of elements in this tensor.
     * <p>
     * This is equal to the product of all dimension sizes.
     * For example, a 2x3x4 tensor contains 24 elements.
     *
     * @return total number of elements
     */
    int elements();

    /**
     * Returns the linear index of the maximum value in this tensor.
     * <p>
     * For multi-dimensional tensors, returns the index in flattened (row-major)
     * order. For example, in a 2x3 matrix, position (1,2) would return index 5.
     *
     * @return linear index of maximum element
     */
    int argmax();

    /**
     * Moves this tensor to the specified compute device.
     * <p>
     * If the tensor is already on the target device, returns this tensor.
     * Otherwise creates a new tensor on the target device with a copy of the data.
     *
     * @param device target device (Device, SiliconDevice, or null for CPU)
     * @return tensor on the target device
     */
    Tensor to(Object device);

    /**
     * Creates a deep copy of this tensor.
     * <p>
     * The clone has its own copy of the data buffer and shape information.
     * Modifying the clone will not affect the original tensor.
     *
     * @implNote the cloned tensor will not have an active autograd context
     * @return new independent copy of this tensor
     */
    Tensor clone();

    /**
     * Adds this tensor with another tensor element-wise.
     * @param other the tensor to add
     * @return the current tensor modified
     */
    Tensor add(Tensor other);
    
    /**
     * Performs element-wise addition of two tensors (alias for `add`).
     * @param other the tensor to add
     * @return a new tensor with the result
     */
    default Tensor plus(Tensor other) {
        return clone().add(other);
    }
    
    /**
     * Adds this tensor with a constant value element-wise.
     * @param value the constant value to add
     * @return the current tensor modified
     */
    Tensor add(double value);

    /**
     * Adds a constant value to this tensor element-wise (alias for `add`).
     * @param value the constant value to add
     * @return a new tensor with the result
     */
    default Tensor plus(double value) {
        return clone().add(value);
    }

    /**
     * Subtracts another tensor from this tensor element-wise.
     * @param other the tensor to subtract
     * @return the current tensor modified
     */
    Tensor sub(Tensor other);
    
    /**
     * Performs element-wise subtraction of two tensors (alias for `sub`).
     * @param other the tensor to subtract
     * @return a new tensor with the result
     */
    default Tensor minus(Tensor other) {
        return clone().sub(other);
    }
    
    /**
     * Subtracts a constant value from this tensor element-wise.
     * @param value the constant value to subtract
     * @return the current tensor modified
     */
    Tensor sub(double value);

    /**
     * Subtracts a constant value from this tensor element-wise (alias for `sub`).
     * @param value the constant value to subtract
     * @return a new tensor with the result
     */
    default Tensor minus(double value) {
        return clone().sub(value);
    }

    /**
     * Multiplies this tensor with another tensor element-wise.
     * @param other the tensor to multiply
     * @return the current tensor modified
     */
    Tensor mul(Tensor other);
    
    /**
     * Performs element-wise multiplication of two tensors (alias for `mul`).
     * @param other the tensor to multiply
     * @return a new tensor with the result
     */
    default Tensor times(Tensor other) {
        return clone().mul(other);
    }
    
    /**
     * Multiplies this tensor with a constant value element-wise.
     * @param value the constant value to multiply
     * @return a new tensor with the result
     */
    Tensor mul(double value);
    
    /**
     * Multiplies a constant value with this tensor element-wise (alias for `mul`).
     * @param value the constant value to multiply
     * @return a new tensor with the result
     */
    default Tensor times(double value) {
        return clone().mul(value);
    }

    /**
     * Divides this tensor by another tensor element-wise.
     * @param other the tensor to divide by
     * @return the current tensor modified
     */
    Tensor div(Tensor other);
    
    /**
     * Performs element-wise division of two tensors (alias for `div`).
     * @param other the tensor to divide by
     * @return a new tensor with the result
     */
    default Tensor divide(Tensor other) {
        return clone().div(other);
    }

    /**
     * Divides this tensor by a constant value element-wise.
     * @param value the constant value to divide by
     * @return the current tensor modified
     */
    Tensor div(double value);

    /**
     * Divides this tensor by a constant value element-wise (alias for `div`).
     * @param value the constant value to divide by
     * @return a new tensor with the result
     */
    default Tensor divide(double value) {
        return clone().div(value);
    }

    /**
     * Raises each element of the tensor to the power of the given value.
     * @param value the exponent
     * @return a new tensor with the result
     */
    Tensor pow(double value);

    /**
     * Raises each element of the tensor to the power of the corresponding element in another tensor.
     * @param other the tensor containing the exponents
     * @return a new tensor with the result
     */
    Tensor pow(Tensor other);

    /**
     * Takes the square root of each element of the tensor.
     * @return a new tensor with the result
     */
    Tensor sqrt();

    /**
     * Flattens the tensor in a 1D vector.
     * @return a new tensor with shape: [elements]
     */
    Tensor flatten();

    /**
     * Computes the (batched) matrix product of this tensor by another.
     * <p>
     * Treats the last two dimensions of each tensor as matrices of shape
     * {@code [m × n]} and {@code [n × p]}. Any preceding dimensions are
     * treated as batch dimensions and must be broadcastable between the two
     * tensors according to broadcasting rules.
     * <p>
     * Formally, if this tensor has shape
     * {@code [..., m, n]} and {@code other} has shape {@code [..., n, p]}, then
     * the result will have shape {@code [..., m, p]}.
     * <p>
     * <strong>For example:</strong>
     * <blockquote><pre>
     *     // A has shape [2, 3], B has shape [3, 4]
     *     A.matmul(B)      // returns shape [2, 4]
     *
     *     // A has shape [5, 2, 3], B has shape [5, 3, 4]
     *     A.matmul(B)      // returns shape [5, 2, 4]
     *
     *     // A has shape [5, 1, 2, 3], B has shape [1, 6, 3, 4]
     *     A.matmul(B)      // returns shape [5, 6, 2, 4] (batch dims broadcasted)
     * </pre></blockquote>
     *
     * @param other the right-hand operand
     * @return a new tensor containing the matrix product, with shape {@code [..., m, p]}
     * @throws IllegalArgumentException if the two tensors’ shapes are not compatible for matrix multiplication
     * @see #transpose()
     */
    Tensor matmul(Tensor other);

    /**
     * Computes a convolution between this tensor and the specified kernel.
     * <p>
     * Convolution works through the usage of Im2Col, supports multiple dimensions and broadcasting.
     * <p>
     * Formally, if this tensor has shape {@code [..., channels, h, w]} and kernel has shape {@code [..., channels, h, w]}, then
     * the result will have shape {@code [..., h_out, w_out]}.
     * <strong>For example:</strong>
     * <blockquote><pre>
     *     // A has shape [3, 15, 15], B has shape [3, 4, 4]
     *     A.convolve(B); // returns shape [12, 12]
     *
     *     // A has shape [16, 3, 64, 64], B has shape [3, 7, 7]
     *     A.convolve(B); // returns shape [16, 58, 58]
     *
     *     // A has shape [16, 3, 64, 64], B has shape [32, 7, 7]
     *     A.convolve(B); // returns [16, 32, 58, 58]
     * </pre></blockquote>
     *
     * @param kernel the kernel tensor to use for convolution.
     * @return a new tensor resulting from the convolution.
     * @throws IllegalArgumentException if tensor dimensions are not compatible.
     */
    Tensor convolve(Tensor kernel);

    /**
     * Computes a convolution between this tensor and the specified kernel with a configurable stride.
     *
     * @param kernel the kernel tensor to use for convolution
     * @param stride the stride to apply on both spatial dimensions
     * @return a new tensor resulting from the convolution
     */
    Tensor convolve(Tensor kernel, int stride);

    /**
     * Performs a layer normalization along this tensor.
     * @param epsilon the epsilon to avoid division by zero
     * @return the current tensor
     */
    Tensor layerNorm(double epsilon);

    /**
     * Computes the Euclidean distance between this tensor and the given one.
     * Both tensors must have the same shape.
     *
     * @param other the tensor to compare against
     * @return the Euclidean distance as a double
     * @throws IllegalArgumentException if shapes do not match
     */
    double distance(Tensor other);

    /**
     * Computes the squared Euclidean distance between this tensor and the given one.
     * Both tensors must have the same shape.
     *
     * @param other the tensor to compare against
     * @return the Euclidean distance as a double
     * @throws IllegalArgumentException if shapes do not match
     */
    double distanceSquared(Tensor other);
    
    /**
     * Squeezes the tensor by removing all dimensions which are equal to one.
     * @return the squeezed tensor
     */
    Tensor squeeze();

    /**
     * Squeezes the tensor by removing the select dimension if it's equal to one
     * @param dimension the dimension to remove
     * @return the squeezed tensor
     */
    Tensor squeeze(int dimension);

    /**
     * Unsqueezes the tensor by adding a dimension with one at the start.
     * @return the unsqueezed tensor
     */
    Tensor unsqueeze();

    /**
     * Broadcasts this tensor to match another shape.
     * @param targetShape the wanted shape
     * @return a copy of the current tensor, with the target shape
     */
    Tensor broadcast(int[] targetShape);

    /**
     * Broadcasts this tensor to match the shape of the other tensor.
     * @param other the other tensor
     * @return a copy of the current tensor, with the same shape as the other one
     */
    Tensor broadcastLike(Tensor other);

    /**
     * Unsqueezes the tensor by adding a dimension with one at the specified dimension
     * @param dimension the dimension index
     * @return the unsqueezed tensor
     */
    Tensor unsqueeze(int dimension);
    
    /**
     * Computes a lazy-transposition of this tensor.
     * This operation has complexity O(1) if the normal matrix multiplication
     * provider is used. When SIMD is enabled, this operation delegates to
     * a high-performance in place transposition.
     * @return a new transposed tensor.
     */
    Tensor transpose();
    
    /**
     * Computes a lazy-transposition of this tensor.
     * This operation has complexity O(1) if the normal matrix multiplication
     * provider is used. When SIMD is enabled, this operation delegates to
     * a high-performance in place transposition.
     * @param dim1 the first dimension to transpose
     * @param dim2 the second dimension to transpose
     * @return a new transposed tensor.
     */
    Tensor transpose(int dim1, int dim2);

    /**
     * Gets whether the current tensor is transposed.
     * @return true if the tensor is transposed, false otherwise
     */
    boolean transposed();
    /**
     * Computes and returns the sum of all elements in the tensor.
     * @return the sum of all values in the tensor as a double
     */
    double sum();

    /**
     * Computes and returns the mean (average) of all elements in the tensor.
     * @return the mean of all values in the tensor as a double
     */
    double mean();

    /**
     * Computes and returns the variance of all elements in the tensor.
     * <p>
     * Variance is calculated as the average of the squared deviations from the mean.
     *
     * @return the variance of all values in the tensor as a double
     */
    double variance();

    /**
     * Returns the maximum value among all elements in the tensor.
     * <p>
     * If the tensor is empty, this method returns Double.NEGATIVE_INFINITY.
     *
     * @return the maximum value in the tensor as a double
     */
    double max();

    /**
     * Returns the minimum value among all elements in the tensor.
     * <p>
     * If the tensor is empty, this method returns Double.POSITIVE_INFINITY.
     *
     * @return the minimum value in the tensor as a double
     */
    double min();

    /**
     * Computes the sum of elements along the specified dimension.
     *
     * @param dim the dimension along which to sum the elements, -1 to specify the last dimension
     * @param keepDim if true, retains the reduced dimension with size 1; otherwise, the dimension is removed
     * @return a new tensor containing the sum along the specified dimension
     *
     * @apiNote the returned tensor preserves the autograd context of the input
     */
    Tensor sum(int dim, boolean keepDim);

    /**
     * Computes the mean of elements along the specified dimension.
     *
     * @param dim the dimension along which to compute the mean, -1 to specify the last dimension
     * @param keepDim if true, retains the reduced dimension with size 1; otherwise, the dimension is removed
     * @return a new tensor containing the mean along the specified dimension
     *
     * @apiNote the returned tensor preserves the autograd context of the input
     */
    Tensor mean(int dim, boolean keepDim);
    
    /**
     * Computes the variance of elements along the specified dimension.
     * @param dim the dimension along which to compute the mean, -1 to specify the last dimension
     * @param keepDim if true, retains the reduced dimension with size 1; otherwise, the dimension is removed
     * @return a new tensor containing the variance along the specified dimension
     */
    Tensor variance(int dim, boolean keepDim);
    
    /**
     * Computes the variance of elements along the specified dimension with a specified mean tensor.
     * @param mean a tensor representing the mean of this tensor.
     * @param dim the dimension along which to compute the mean, -1 to specify the last dimension
     * @param keepDim if true, retains the reduced dimension with size 1; otherwise, the dimension is removed
     * @return a new tensor containing the variance along the specified dimension
     */
    Tensor variance(Tensor mean, int dim, boolean keepDim);

    /**
     * Computes the sign of each element in the tensor.
     * <p>
     * The sign is defined as: -1 for negative values, 0 for zero, and 1 for positive values.
     *
     * @return a new tensor with the sign of each element
     */
    Tensor sign();

    /**
     * Reshapes the current tensor to a new shape.
     * @param newShape the new shape of the tensor
     * @return a copy of the tensor with a new shape
     */
    Tensor reshape(int... newShape);

    /**
     * Concatenates this tensor with another tensor along the last dimension.
     * For this method to work, tensors must have shape <code>[..., a, b]</code> with the same <code>a</code>.
     * @param other the tensor to concatenate
     * @return a new concatenated tensor
     */
    Tensor concat(Tensor other);

    /**
     * Concatenates this tensor with another tensor along the last dimension.
     * For this method to work, tensors must have shape <code>[..., a, b]</code> with the same <code>a</code>.
     * @param other the tensor to concatenate
     * @return a new concatenated tensor
     */
    Tensor concat(Tensor other, int dimension);

    /**
     * Activates all the elements of this tensor using the specified activation function.
     * @param activation the activation function
     * @return a new activated tensor
     */
    Tensor activate(Activation activation);

    /**
     * Selects a sub-tensor from this tensor, given the specified dimension
     * and index.
     *
     * @param dim the dimension to select from
     * @param index the index in the specified dimension to select
     * @return a new tensor with the selected values
     */
    Tensor select(int dim, int index);

    /**
     * Slices the tensor according to the specified ranges for each dimension.
     *
     * @param ranges the ranges specifying the slice for each dimension
     * @return a new tensor containing the sliced data
     * @throws IllegalArgumentException if more ranges are specified than the number of dimensions
     */
    Tensor slice(Range... ranges);

    /**
     * Applies a given function to each element of the tensor and returns a new tensor with the results.
     * @param function the function to apply
     * @return the current tensor
     */
    Tensor map(D2DFunction function);

    /**
     * Sets all elements of this tensor to the given value.
     * @param value the value to fill the tensor with
     * @return the current tensor
     */
    Tensor fill(float value);

    /**
     * Fills the tensor with the values generated by the given supplier.
     * @param supplier the supplier of values
     * @return the current tensor
     */
    Tensor fill(Supplier<Double> supplier);

    /**
     * Gets the autograd context for this tensor.
     * @return the autograd context instance
     */
    AutogradContext getAutogradContext();

    /**
     * Updates the autograd context instance for this tensor.
     * @param autogradContext the new autograd context
     */
    void setAutogradContext(AutogradContext autogradContext);

    /**
     * Enables autograd for this tensor.
     * @return the current tensor
     */
    Tensor withGrad();
    
    /**
     * Disables autograd for this tensor.
     * @return the current tensor
     */
    Tensor noGrad();

    /**
     * Returns whether this tensor uses autograd.
     * @return true if this tensor uses autograd, false otherwise
     */
    boolean usesGrad();

    /**
     * Zeros the gradient for this tensor.
     */
    void zeroGrad();

    /**
     * Gets the gradient for this tensor.
     * @return the gradient of this tensor
     */
    Tensor grad();

    /**
     * Computes the backward pass for this tensor.
     */
    void backward();

    /**
     * Computes the backward pass for this tensor with the specified gradient.
     * @param gradOutput the gradient
     */
    void backward(Tensor gradOutput);

    /**
     * Executes the specified operation on this tensor.
     * @param operation the operation to execute
     * @return the resulting tensor from the operation
     */
    Tensor forward(Operation operation);

    /**
     * Executes the specified operation on this tensor and the specified other tensor.
     * @param operation the operation to execute
     * @param other the other tensor
     * @return the resulting tensor from the operation
     */
    Tensor forward(Operation operation, Tensor other);

    /**
     * Executes the specified operation on this tensor and the specified other tensors.
     * @param operation the operation to execute
     * @param others the other tensors
     * @return the resulting tensor from the operation
     */
    Tensor forward(Operation operation, Tensor... others);

    /**
     * Delegates to {@link #forward(Operation, Tensor)} using {@link AddOperation}
     * @return the resulting tensor from the operation
     */
    Tensor addGrad(Tensor other);

    /**
     * Delegates to {@link #forward(Operation, Tensor)} using {@link MulOperation}
     * @return the resulting tensor from the operation
     */
    Tensor mulGrad(Tensor other);

    /**
     * Delegates to {@link #forward(Operation, Tensor)} using {@link DivOperation}
     * @param other the other tensor
     * @return the resulting tensor from the operation
     */
    Tensor divGrad(Tensor other);

    /**
     * Delegates to {@link #forward(Operation, Tensor)} using {@link SubOperation}
     * @param other the other tensor
     * @return the resulting tensor from the operation
     */
    Tensor subGrad(Tensor other);
    
    /**
     * Delegates to {@link #forward(Operation, Tensor)} using {@link SliceOperation}
     * @param ranges the ranges to slice this tensor on
     * @return the resulting tensor from the operation
     */
    Tensor sliceGrad(Range... ranges);

    /**
     * Delegates to {@link #forward(Operation, Tensor)} using {@link MatMulOperation}.
     * @param other the other tensor
     * @return the resulting tensor from the operation
     */
    Tensor matmulGrad(Tensor other);

    /**
     * Delegates to {@link #forward(Operation, Tensor)} using {@link ConvolveOperation}.
     * @param other the convolution kernel
     * @return the resulting tensor from the operation
     */
    Tensor convolveGrad(Tensor other);

    /**
     * Delegates to {@link #forward(Operation, Tensor)} using {@link ConvolveOperation} with a configurable stride.
     * @param other the convolution kernel
     * @param stride the stride to apply on both spatial dimensions
     * @return the resulting tensor from the operation
     */
    Tensor convolveGrad(Tensor other, int stride);

    /**
     * Delegates to {@link #forward(Operation, Tensor)} using {@link MaxPoolOperation}.
     * @param stride the stride to use on the pooling
     * @param windowHeight the window height of the pooling
     * @param windowWidth the window width of the pooling
     * @return the pooling result in a new tensor
     */
    Tensor maxPoolGrad(int stride, int windowHeight, int windowWidth);

    /**
     * Delegates to {@link #forward(Operation, Tensor)} using {@link TransposeOperation}.
     * @return the resulting tensor from the operation
     */
    Tensor transposeGrad();

    /**
     * Delegates to {@link #forward(Operation, Tensor)} using {@link TransposeOperation}
     * @param dim1 the first dimension to transpose
     * @param dim2 the second dimension to transpose
     * @return the resulting tensor from the operation
     */
    Tensor transposeGrad(int dim1, int dim2);

    /**
     * Delegates to {@link #forward(Operation, Tensor)} using {@link ActivationOperation}
     * @param activation the activation to apply
     * @return the resulting tensor from the operation
     */
    Tensor activateGrad(Activation activation);
    
    /**
     * Delegates to {@link #forward(Operation, Tensor)} using {@link ConcatOperation}
     * @param other the other tensor
     * @return the resulting tensor from the operation
     */
    Tensor concatGrad(Tensor other, int dim);

    /**
     * Delegates to {@link #forward(Operation, Tensor)} using {@link ReshapeOperation}.
     * @param newShape the new shape of this tensor
     * @return the resulting tensor with a new shape
     */
    Tensor reshapeGrad(int... newShape);

    /**
     * Delegates to {@link #forward(Operation, Tensor)} using {@link SqueezeOperation} and no parameters.
     * @return a clone of this tensor with the squeezed shape
     */
    Tensor squeezeGrad();

    /**
     * Delegates to {@link #forward(Operation, Tensor)} using {@link SqueezeOperation} and no parameters.
     * @param dimension the dimension to squeeze on
     * @return a clone of this tensor with the squeezed shape
     */
    Tensor squeezeGrad(int dimension);
    
    /**
     * Flips the tensor by 180 degrees.
     * @return a new flipped tensor
     */
    Tensor flip();

    /**
     * Applies softmax to the tensor with a default temperature of 1.
     * @return a new soft-maxed tensor
     */
    Tensor softmax();

    /**
     * Applies softmax to the tensor with the specified temperature.
     * @param temperature a parameter indicating how much to smooth the distribution
     * @return a new soft-maxed tensor
     */
    Tensor softmax(double temperature);

    /**
     * Gets a string containing all the values of this tensor in the specified format.
     * @param format the string format
     * @return the tensor values on a string
     */
    String toString(String format);

    private Tensor activateAuto(Activation function) {
        return usesGrad() ? activateGrad(function) : activate(function);
    }

    /**
     * Computes the ReLU activation on this tensor.
     * @return a copy of this tensor with the activated values
     */
    default Tensor relu() {
        return activateAuto(new ReLU());
    }

    /**
     * Computes the Sigmoid activation on this tensor.
     * @return a copy of this tensor with the activated values
     */
    default Tensor sigmoid() {
        return activateAuto(new Sigmoid());
    }

    /**
     * Computes the Tanh activation on this tensor.
     * @return a copy of this tensor with the activated values
     */
    default Tensor tanh() {
        return activateAuto(new Tanh());
    }
    
    default Tensor cpu() {
        return to(null);
    }
}
