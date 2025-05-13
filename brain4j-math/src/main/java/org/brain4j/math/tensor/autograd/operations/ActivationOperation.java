package org.brain4j.math.tensor.autograd.operations;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.autograd.Operation;

import java.util.Arrays;

public class ActivationOperation implements Operation {

    private final Activation activation;

    public ActivationOperation(Activation activation) {
        this.activation = activation;
    }

    @Override
    public Tensor forward(Tensor... inputs) {
        System.out.println("In shape: " + Arrays.toString(inputs[0].shape()));
        return activation.activate(inputs[0]);
    }

    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor... inputs) {
        Tensor input = inputs[0]; // [batch_size, input_size]
        Tensor derivative = activation.getDerivative(input); // ∂activation/∂x

        Tensor gradInput = gradOutput.mul(derivative); // Chain rule: dL/dx = dL/dy * dy/dx

        return new Tensor[] { gradInput };
    }
}
