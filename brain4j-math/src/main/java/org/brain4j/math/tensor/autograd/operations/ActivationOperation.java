package org.brain4j.math.tensor.autograd.operations;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;
import org.brain4j.math.tensor.autograd.Operation;

import java.util.Arrays;

public class ActivationOperation implements Operation {

    private final Activation activation;

    public ActivationOperation(Activation activation) {
        this.activation = activation;
    }

    @Override
    public Tensor forward(Tensor... inputs) {
        return activation.activate(inputs[0]);
    }

    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor... inputs) {
        Tensor activated = activation.activate(inputs[0]);
        Tensor derivative = activation.getDerivative(activated); // ∂activation/∂x
        Tensor gradInput = gradOutput.times(derivative); // Chain rule: dL/dx = dL/dy * dy/dx

        return new Tensor[] { gradInput };
    }
}
