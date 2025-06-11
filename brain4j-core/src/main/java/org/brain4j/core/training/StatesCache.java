package org.brain4j.core.training;

import org.brain4j.core.model.Model;
import org.brain4j.math.tensor.Tensor;

public class StatesCache {

    private final Tensor[] preActivations;
    private final Tensor[] hiddenStates;

    public StatesCache(Model model) {
        int size = model.flattened().size();
        this.preActivations = new Tensor[size];
        this.hiddenStates = new Tensor[size];
    }

    public Tensor preActivation(int index) {
        return preActivations[index];
    }

    public void setPreActivation(int index, Tensor preActivation) {
        preActivations[index] = preActivation;
    }

    public Tensor hiddenState(int index) {
        return hiddenStates[index];
    }

    public void setHiddenState(int index, Tensor hidden) {
        hiddenStates[index] = hidden;
    }
}

