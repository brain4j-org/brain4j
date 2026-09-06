package org.brain4j.math.tensor.autograd;

import org.brain4j.math.tensor.Tensor;

public class AutogradContext {

    private final Tensor owner;
    private final boolean requiresGrad;
    private Tensor[] inputs;
    private Tensor grad;
    private Operation operation;
    private int expectedContributions = 0;
    private int receivedContributions = 0;

    public AutogradContext(Tensor owner, boolean requiresGrad) {
        this.owner = owner;
        this.requiresGrad = requiresGrad;
        this.grad = null;
    }

    public void zeroGrad() {
        this.grad = null;
        this.receivedContributions = 0;
        this.expectedContributions = 0;
    }

    public void setOperation(Operation operation, Tensor... inputs) {
        this.operation = operation;
        this.inputs = inputs;

        if (inputs == null) return;

        for (Tensor input : inputs) {
            if (input == null || !input.usesGrad()) continue;

            input.getAutogradContext().increaseContributes();
        }
    }

    public void increaseContributes() {
        expectedContributions++;
    }

    public boolean requiresGrad() {
        return requiresGrad;
    }
    
    public Tensor getGrad() {
        return grad;
    }
    
    public void backward(Tensor gradOutput) {
        if (!requiresGrad) return;
        
        this.grad = grad == null ? gradOutput.copy() : grad.add(gradOutput.broadcastLike(grad));

        receivedContributions++;

        int needed = Math.max(1, expectedContributions);

        if (receivedContributions < needed) return;

        if (operation == null) return;

        Tensor[] inputGrads = operation.backward(grad, owner, inputs);

        for (int i = 0; i < inputs.length; i++) {
            Tensor input = inputs[i];

            if (input == null || !input.usesGrad()) continue;

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