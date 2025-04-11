package net.echo.math4j.math.tensor.autograd;

import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.impl.TensorCPU;

public class AutogradContext {

    private final boolean requiresGrad;
    private Tensor[] inputs;
    private Tensor grad;
    private Operation operation;

    public AutogradContext(boolean requiresGrad) {
        this.requiresGrad = requiresGrad;
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
            int[] shape = this.inputs[0].shape();
            grad = new TensorCPU(shape);
        }

        return grad;
    }
    
    public void backward(Tensor gradOutput) {
        if (!requiresGrad) return;

        grad = grad == null ? gradOutput.clone() : grad.plus(gradOutput);

        if (operation == null) return;

        Tensor[] inputGrads = operation.backward(gradOutput, inputs);

        for (int i = 0; i < inputs.length; i++) {
            Tensor input = inputs[i];

            if (!input.requiresGrad()) continue;

            input.backward(inputGrads[i]);
        }
    }
} 