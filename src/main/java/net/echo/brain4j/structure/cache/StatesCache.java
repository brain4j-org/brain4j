package net.echo.brain4j.structure.cache;

import net.echo.brain4j.convolution.Kernel;
import net.echo.brain4j.layer.impl.convolution.ConvLayer;
import net.echo.brain4j.structure.Neuron;

public class StatesCache {

    private final Kernel[] featureMaps;
    private final float[] valuesCache;
    private final float[] deltasCache;

    public StatesCache() {
        this.valuesCache = new float[Parameters.TOTAL_NEURONS];
        this.deltasCache = new float[Parameters.TOTAL_NEURONS];
        this.featureMaps = new Kernel[Parameters.TOTAL_CONV_LAYER];
    }

    public void setFeatureMap(ConvLayer layer, Kernel output) {
        featureMaps[layer.getId()] = output;
    }

    public Kernel getFeatureMap(ConvLayer layer) {
        return featureMaps[layer.getId()];
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
}
