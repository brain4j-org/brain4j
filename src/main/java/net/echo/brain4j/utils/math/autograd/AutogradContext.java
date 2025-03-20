package net.echo.brain4j.utils.math.autograd;

import net.echo.brain4j.utils.math.tensor.Tensor;

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
            grad = Tensor.zeros(shape);
        }
        return grad;
    }
    
    public void backward(Tensor gradOutput) {
        if (!requiresGrad) {
            return;
        }
        
        if (grad == null) {
            grad = gradOutput.clone();
        } else {
            grad = grad.plus(gradOutput);
        }
        
        if (operation != null) {
            Tensor[] inputGrads = operation.backward(gradOutput, inputs);

            for (int i = 0; i < inputs.length; i++) {
                if (inputs[i].requiresGrad()) {
                    inputs[i].backward(inputGrads[i]);
                }
            }
        }
    }
} 