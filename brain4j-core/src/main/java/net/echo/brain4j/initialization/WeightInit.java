package net.echo.brain4j.initialization;

import net.echo.brain4j.initialization.impl.*;

/**
 * Enum that defines the different types of weight initialization strategies used for neural networks.
 * Each type corresponds to a different algorithm that initializes the weights of a neural network layer.
 */
public enum WeightInit {

    /**
     * Normal initialization uses a normal distribution to initialize the weights.
     * It typically uses a mean of 0 and a small standard deviation.
     */
    NORMAL(new NormalInit()),

    /**
     * He initialization (also known as Kaiming initialization) is designed for layers with ReLU activations.
     * It initializes weights with a normal distribution, scaled by the square root of 2 divided by the number of input neurons.
     */
    HE(new HeInit()),

    /**
     * Uniform Xavier initialization is specifically designed for layers with sigmoid or tanh activations.
     * It initializes weights using a uniform distribution,
     * scaled by the square root of 6 divided by the sum of the number of input and output neurons.
     */
    UNIFORM_XAVIER(new UniformXavierInit()),

    /**
     * Normal Xavier initialization is specifically designed for layers with sigmoid or tanh activations.
     * It initializes weights using a normal distribution,
     * scaled by the square root of 2 divided by the number of input and output neurons.
     */
    NORMAL_XAVIER(new NormalXavierInit()),

    /**
     * LeCun initialization is specifically designed for layers with the sigmoid or tanh activation functions.
     * It initializes weights using a normal distribution, scaled by 1 divided by the number of input neurons.
     */
    LECUN(new LeCunInit());

    private final WeightInitializer initializer;

    /**
     * Constructor for the enum. Associates each type with a corresponding weight initializer.
     *
     * @param initializer the weight initializer for the corresponding type
     */
    WeightInit(WeightInitializer initializer) {
        this.initializer = initializer;
    }

    /**
     * Gets the weight initializer associated with the initialization type.
     *
     * @return the WeightInitializer instance for the current initialization type
     */
    public WeightInitializer getFunction() {
        return initializer;
    }
}