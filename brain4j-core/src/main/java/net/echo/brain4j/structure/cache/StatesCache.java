package net.echo.brain4j.structure.cache;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.Neuron;
import net.echo.math4j.math.tensor.Tensor;

public class StatesCache {

    private final Tensor[] inputTensorsCache;
    private final Tensor[] outputTensorsCache;

    private final float[] valuesCache;
    private final float[] deltasCache;

    public StatesCache() {
        this.inputTensorsCache = new Tensor[Parameters.TOTAL_LAYERS];
        this.outputTensorsCache = new Tensor[Parameters.TOTAL_LAYERS];

        this.valuesCache = new float[Parameters.TOTAL_NEURONS];
        this.deltasCache = new float[Parameters.TOTAL_NEURONS];
    }

    public float getValue(Neuron neuron) {
        return valuesCache[neuron.getId()];
    }

    public float getDelta(Neuron neuron) {
        return deltasCache[neuron.getId()];
    }

    public void setInputTensor(Layer<?, ?> layer, Tensor value) {
        inputTensorsCache[layer.getId()] = value;
    }

    public Tensor getInputTensor(Layer<?, ?> layer) {
        return inputTensorsCache[layer.getId()];
    }

    public void setOutputTensor(Layer<?, ?> layer, Tensor value) {
        outputTensorsCache[layer.getId()] = value;
    }

    public Tensor getOutputTensor(Layer<?, ?> layer) {
        return outputTensorsCache[layer.getId()];
    }
}
