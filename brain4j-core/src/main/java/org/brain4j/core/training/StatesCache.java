package org.brain4j.core.training;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.tensor.Tensor;

import java.util.HashMap;
import java.util.Map;

public class StatesCache {

    private final Map<Layer, Tensor> preActivations;
    private final Map<Layer, Tensor> hiddenStates;

    public StatesCache() {
        this.preActivations = new HashMap<>();
        this.hiddenStates = new HashMap<>();
    }

    public Tensor preActivation(Layer layer) {
        return preActivations.get(layer);
    }

    public void setPreActivation(Layer layer, Tensor preActivation) {
        preActivations.put(layer, preActivation);
    }

    public Tensor hiddenState(Layer layer) {
        return hiddenStates.get(layer);
    }

    public void setHiddenState(Layer layer, Tensor hidden) {
        hiddenStates.put(layer, hidden);
    }
}

