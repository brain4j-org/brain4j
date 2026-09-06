package org.brain4j.math.data;

import org.brain4j.math.Copyable;
import org.brain4j.math.tensor.Tensor;

import java.util.Arrays;

public record Sample(Tensor[] inputs, Tensor[] labels) implements Copyable<Sample> {
    
    public Sample(Tensor input, Tensor label) {
        this(new Tensor[] { input }, new Tensor[] { label });
    }
    
    public Tensor getInput(int index) {
        return inputs[index];
    }
    
    public Tensor getLabel(int index) {
        return labels[index];
    }
    
    @Override
    public Sample copy() {
        Tensor[] clonedInputs = new Tensor[inputs.length];
        Tensor[] clonedLabels = new Tensor[labels.length];
        
        for (int i = 0; i < clonedInputs.length; i++) {
            clonedInputs[i] = inputs[i].copy();
        }
        
        for (int i = 0; i < clonedLabels.length; i++) {
            clonedLabels[i] = labels[i].copy();
        }
        
        return new Sample(clonedInputs, clonedLabels);
    }
}
