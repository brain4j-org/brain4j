package org.brain4j.core.training;

import org.brain4j.core.model.Model;
import org.brain4j.math.tensor.Tensor;

public class StatesCache {

    private final Tensor[] inputs;
    private final Tensor[] outputs;
    private final Tensor[] preActivations;
    private final Tensor[] hiddenStates;

    public StatesCache(Model model) {
        this.inputs = new Tensor[model.size()];
        this.outputs = new Tensor[model.size()];
        this.preActivations = new Tensor[model.size()];
        this.hiddenStates = new Tensor[model.size()];
    }

    public Tensor input(int index) {
        return inputs[index];
    }

    public void setInput(int index, Tensor input) {
        inputs[index] = input;
    }

    public Tensor output(int index) {
        return outputs[index];
    }

    public void setOutput(int index, Tensor output) {
        outputs[index] = output;
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

