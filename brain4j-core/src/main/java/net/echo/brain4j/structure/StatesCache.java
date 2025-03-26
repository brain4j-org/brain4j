package net.echo.brain4j.structure;

import net.echo.brain4j.layer.Layer;
import net.echo.math4j.math.tensor.Tensor;

public class StatesCache {

    private final Tensor[] inputTensorsCache;
    private final Tensor[] outputTensorsCache;

    public StatesCache() {
        this.inputTensorsCache = new Tensor[Parameters.TOTAL_LAYERS];
        this.outputTensorsCache = new Tensor[Parameters.TOTAL_LAYERS];
    }

    public void setInputTensor(Layer layer, Tensor value) {
        inputTensorsCache[layer.getId()] = value;
    }

    public Tensor getInputTensor(Layer layer) {
        return inputTensorsCache[layer.getId()];
    }

    public void setOutputTensor(Layer layer, Tensor value) {
        outputTensorsCache[layer.getId()] = value;
    }

    public Tensor getOutputTensor(Layer layer) {
        return outputTensorsCache[layer.getId()];
    }
}
