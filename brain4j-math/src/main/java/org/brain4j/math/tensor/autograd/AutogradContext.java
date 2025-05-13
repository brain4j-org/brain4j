package org.brain4j.math.tensor.autograd;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.TensorCPU;

import java.util.Arrays;

public class AutogradContext {

    private final boolean requiresGrad;
    private Tensor[] inputs;
    private Tensor grad;
    private Operation operation;

    public AutogradContext(boolean requiresGrad) {
        this.requiresGrad = requiresGrad;
        this.grad = null;
    }

    public void zerograd() {
        this.grad = null;
    }

    public void setOperation(Operation operation, Tensor... inputs) {
        this.operation = operation;
        this.inputs = inputs;
    }
    
    public boolean requiresGrad() {
        return requiresGrad;
    }
    
    public Tensor getGrad() {
        if (grad == null) {
            int[] shape = inputs[0].shape();
            this.grad = new TensorCPU(shape);
        }

        return grad;
    }
    
    public void backward(Tensor gradOutput) {
        if (!requiresGrad) return;

//        System.out.println("Gotten grad output = " + gradOutput);
        this.grad = grad == null ? gradOutput.clone() : grad.plus(gradOutput);

        if (operation == null) return;

//        System.out.println("---------------------------------");
//        System.out.println("Launched backward pass for operation: " + operation.getClass().getSimpleName());
//        System.out.println("Gradient output shape: " + Arrays.toString(gradOutput.shape()));
//        System.out.println("Gradient output: " + gradOutput);

        Tensor[] inputGrads = operation.backward(gradOutput, inputs);

        for (int i = 0; i < inputs.length; i++) {
            Tensor input = inputs[i];

            if (!input.usesGrad()) continue;

            input.backward(inputGrads[i]);
        }
    }

    public Tensor[] inputs() {
        return inputs;
    }

    public Operation operation() {
        return operation;
    }
}