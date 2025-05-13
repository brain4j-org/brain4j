package org.brain4j.core.training;

import org.brain4j.core.model.Model;
import org.brain4j.math.tensor.Tensor;

public class StatesCache {

    private final Tensor[] inputs;
    private final Tensor[] outputs;

    public StatesCache(Model model) {
        this.inputs = new Tensor[model.size()];
        this.outputs = new Tensor[model.size()];
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
}
