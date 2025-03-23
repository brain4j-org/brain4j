package net.echo.brain4j.structure.cache;

import net.echo.brain4j.convolution.Kernel;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.Neuron;
import net.echo.math4j.math.tensor.Tensor;

public class StatesCache {

    private final Tensor[] tensorsCache;
    private final Kernel[] inputMap;
    private final Kernel[] featureMaps;
    private final Kernel[] deltaMap;
    private final float[] valuesCache;
    private final float[] deltasCache;

    public StatesCache() {
        this.tensorsCache = new Tensor[Parameters.TOTAL_LAYERS];

        this.valuesCache = new float[Parameters.TOTAL_NEURONS];
        this.deltasCache = new float[Parameters.TOTAL_NEURONS];

        this.inputMap = new Kernel[Parameters.TOTAL_CONV_LAYER];
        this.featureMaps = new Kernel[Parameters.TOTAL_CONV_LAYER];
        this.deltaMap = new Kernel[Parameters.TOTAL_CONV_LAYER];
    }

    public void setInput(Layer<Kernel, Kernel> layer, Kernel input) {
        inputMap[layer.getId()] = input;
    }

    public Kernel getInput(Layer<Kernel, Kernel> layer) {
        return inputMap[layer.getId()];
    }

    public void setFeatureMap(Layer<Kernel, Kernel> layer, Kernel output) {
        featureMaps[layer.getId()] = output;
    }

    public Kernel getFeatureMap(Layer<Kernel, Kernel> layer) {
        return featureMaps[layer.getId()];
    }

    public Kernel getDelta(Layer<Kernel, Kernel> layer) {
        return deltaMap[layer.getId()];
    }

    public void setDelta(Layer<Kernel, Kernel> layer, Kernel delta) {
        deltaMap[layer.getId()] = delta;
    }

    public float getValue(Neuron neuron) {
        return valuesCache[neuron.getId()];
    }

    public float getDelta(Neuron neuron) {
        return deltasCache[neuron.getId()];
    }

    public void setValue(Neuron neuron, float value) {
        valuesCache[neuron.getId()] = value;
    }

    public void setDelta(Neuron neuron, float delta) {
        deltasCache[neuron.getId()] = delta;
    }

    public void addDelta(Neuron neuron, float delta) {
        deltasCache[neuron.getId()] += delta;
    }

    public void setTensor(Layer<?, ?> layer, Tensor value) {
        tensorsCache[layer.getId()] = value;
    }

    public Tensor getTensor(Layer<?, ?> layer) {
        return tensorsCache[layer.getId()];
    }
}
