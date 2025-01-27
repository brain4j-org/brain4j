package net.echo.brain4j.structure;

public class StatesCache {

    public double[] valuesCache;
    public double[] deltasCache;

    public StatesCache() {
        valuesCache = new double[Neuron.NEURON_COUNTER];
        deltasCache = new double[Neuron.NEURON_COUNTER];
    }

    public double getValue(Neuron neuron) {
        return valuesCache[neuron.getId()];
    }

    public double getDelta(Neuron neuron) {
        return deltasCache[neuron.getId()];
    }

    public void setValue(Neuron neuron, double value) {
        valuesCache[neuron.getId()] = value;
    }

    public void setDelta(Neuron neuron, double delta) {
        deltasCache[neuron.getId()] = delta;
    }

    public void addDelta(Neuron neuron, double delta) {
        deltasCache[neuron.getId()] += delta;
    }
}
