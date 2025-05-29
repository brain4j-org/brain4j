package org.brain4j.math.tensor.autograd.operations;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.autograd.Operation;

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

//        System.out.println("inputs[0]:");
//        System.out.println(inputs[0].toString("%.2f"));
//
//        System.out.println("activation type: " + activation.getClass().getSimpleName());
//
//        System.out.println("Activation gradient:");
//        System.out.println(gradInput.toString("%.2f"));
//
//        System.out.println("Gradoutput:");
//        System.out.println(gradOutput.toString("%.2f"));
//
//        System.out.println("Derivative:");
//        System.out.println(derivative.toString("%.2f"));

        return new Tensor[] { gradInput };
    }
}
