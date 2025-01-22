package net.echo.brain4j.threading;

import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.Synapse;

public class NeuronCacheHolder {

    private final double[] gradients;
    private final double[] valuesCache;
    private final double[] deltasCache;

    public NeuronCacheHolder() {
        this.gradients = new double[Synapse.SYNAPSE_COUNTER];
        this.valuesCache = new double[Neuron.NEURON_COUNTER];
        this.deltasCache = new double[Neuron.NEURON_COUNTER];
    }

    public double[] getGradients() {
        return gradients;
    }

    public double getGradient(int index) {
        return gradients[index];
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

    public void setGradient(int index, double gradient) {
        gradients[index] = gradient;
    }

    public void addDelta(Neuron neuron, double delta) {
        deltasCache[neuron.getId()] += delta;
    }

    public void addGradient(int index, double change) {
        gradients[index] += change;
    }
}
