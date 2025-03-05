package net.echo.brain4j.structure.cache;

import net.echo.brain4j.structure.Neuron;

public class StatesCache {

    private final float[] valuesCache;
    private final float[] deltasCache;

    public StatesCache() {
        this.valuesCache = new float[Parameters.TOTAL_NEURONS];
        this.deltasCache = new float[Parameters.TOTAL_NEURONS];
    }

    public double getValue(Neuron neuron) {
        return valuesCache[neuron.getId()];
    }

    public double getDelta(Neuron neuron) {
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
