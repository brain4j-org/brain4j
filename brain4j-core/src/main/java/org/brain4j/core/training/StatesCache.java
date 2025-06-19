package org.brain4j.core.training;

import org.brain4j.core.Brain4J;
import org.brain4j.core.layer.Layer;
import org.brain4j.math.device.Device;
import org.brain4j.math.tensor.Tensor;
import org.jocl.cl_command_queue;

import java.util.HashMap;
import java.util.Map;

public class StatesCache {

    private final Map<Layer, Tensor> preActivations;
    private final Map<Layer, Tensor> hiddenStates;
    private cl_command_queue commandQueue;

    public StatesCache() {
        this.preActivations = new HashMap<>();
        this.hiddenStates = new HashMap<>();

        Device device = Brain4J.currentDevice();

        if (device != null) {
            this.commandQueue = device.newCommandQueue();
        }
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

    public cl_command_queue commandQueue() {
        return commandQueue;
    }
}

