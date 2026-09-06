package org.brain4j.math.activation;

import org.brain4j.math.activation.impl.*;

/**
 * Enum that defines the different types of activation functions used for neural networks.
 * Activation functions are used to introduce non-linearity and therefore allow the network to learn complex patterns.
 */
public enum Activations {

    /**
     * Linear activation is a simple activation function that does not introduce non-linearity,
     * and it's commonly used to project inputs into a linear space.
     */
    LINEAR(new Linear()),

    /**
     * ELU (Exponential Linear Unit) is a non-linear activation function that is similar to ReLU, but allows negative values.
     * It's defined as follows:
     * <blockquote><pre>
     * f(x) = alpha * (exp(x) - 1)
     * </pre></blockquote>
     */
    ELU(new ELU()),

    /**
     * ReLU (Rectified Linear Unit) is a non-linear activation function that sets all negative values to zero.
     * It's defined as follows:
     * <blockquote><pre>
     * f(x) = max(0, x)
     * </pre></blockquote>
     */
    RELU(new ReLU()),

    /**
     * GELU (Gaussian Error Linear Unit) is a non-linear activation function that is similar to ReLU, but allows negative values.
     * It's defined as follows:
     * <blockquote><pre>
     * f(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
     * </pre></blockquote>
     */
    GELU(new GELU()),

    /**
     * Leaky ReLU is a non-linear activation function that allows negative values, but with a small slope.
     * It's defined as follows:
     * <blockquote><pre>
     * f(x) = max(0.01 * x, x)
     * </pre></blockquote>
     */
    LEAKY_RELU(new LeakyReLU()),

    /**
     * Sigmoid is a non-linear activation function that maps values between 0 and 1.
     * It's defined as follows:
     * <blockquote><pre>
     * f(x) = 1 / (1 + exp(-x))
     * </pre></blockquote>
     */
    SIGMOID(new Sigmoid()),

    /**
     * Softmax is a non-linear activation function that maps values between 0 and 1.
     * It's defined as follows:
     * <blockquote><pre>
     * f(x) = exp(x) / sum(exp(x))
     * </pre></blockquote>
     */
    SOFTMAX(new Softmax()),

    /**
     * Tanh is a non-linear activation function that maps values between -1 and 1.
     * It's defined as follows:
     * <blockquote><pre>
     * f(x) = tanh(x)
     * </pre></blockquote>
     */
    TANH(new Tanh()),

    /**
     * Mish is a non-linear activation function that is similar to GELU.
     * It's defined as follows:
     * <blockquote><pre>
     * f(x) = x * tanh(log(1 + exp(x)))
     * </pre></blockquote>
     */
    MISH(new Mish()),

    /**
     * Swish is a non-linear activation function that is similar to GELU.
     * It's defined as follows:
     * <blockquote><pre>
     * f(x) = x * sigmoid(x)
     * </pre></blockquote>
     */
    SWISH(new Swish()),

    /**
     * Softplus is a non-linear activation function that is similar to ReLU.
     * It appears as a smooth version of the ReLU function, and it's definied as follows:
     * <blockquote><pre>
     * f(x) = ln(1 + e^x)
     * </pre></blockquote>
     */
    SOFTPLUS(new SoftPlus());

    private final Activation function;

    /**
     * Constructor for the activations enum. Associates each type with a corresponding activation function.
     * @param function the activation function for the corresponding type
     */
    Activations(Activation function) {
        this.function = function;
    }

    /**
     * Returns the activation function associated with the type.
     * @return The activation function.
     */
    public Activation function() {
        return function;
    }
}
