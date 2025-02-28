package net.echo.brain4j.structure.cache;

import net.echo.brain4j.structure.Neuron;

public class StatesCache {

    private final double[] valuesCache;
    private final double[] deltasCache;

    public StatesCache() {
        this.valuesCache = new double[Parameters.TOTAL_NEURONS];
        this.deltasCache = new double[Parameters.TOTAL_NEURONS];
    }

    public double getValue(Neuron neuron) {
        return valuesCache[neuron.getId()];
    }

    public double getDelta(Neuron neuron) {
        return this.deltasCache[neuron.getId()];
    }

    public void setValue(Neuron neuron, double value) {
        this.valuesCache[neuron.getId()] = value;
    }

    public void setDelta(Neuron neuron, double delta) {
        this.deltasCache[neuron.getId()] = delta;
    }

    public void addDelta(Neuron neuron, double delta) {
        this.deltasCache[neuron.getId()] += delta;
    }
}
