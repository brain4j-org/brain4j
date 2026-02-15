package org.brain4j.math.tensor.autograd.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.autograd.Operation;

public record ActivationOperation(Activation activation) implements Operation {

    @Override
    public int requiredInputs() {
        return 1;
    }
    
    @Override
    public Tensor compute(Tensor... inputs) {
        return activation.activate(inputs[0]);
    }
    
    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor output, Tensor... inputs) {
        Tensor gradInput = activation.derivative(inputs[0], output, gradOutput);
        return new Tensor[] { gradInput };
    }
}
