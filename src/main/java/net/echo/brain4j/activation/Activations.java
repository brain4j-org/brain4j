package net.echo.brain4j.activation;

import net.echo.brain4j.activation.impl.*;

public enum Activations {

    LINEAR(new LinearActivation()),
    ELU(new ELUActivation()),
    RELU(new ReLUActivation()),
    GELU(new GELUActivation()),
    LEAKY_RELU(new LeakyReLUActivation()),
    SIGMOID(new SigmoidActivation()),
    SOFTMAX(new SoftmaxActivation()),
    TANH(new TanhActivation()),
    MISH(new MishActivation()),
    SWISH(new SwishActivation());

    private final Activation function;

    Activations(Activation function) {
        this.function = function;
    }

    public Activation getFunction() {
        return function;
    }
}
