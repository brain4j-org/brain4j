package net.echo.brain4j.structure;

import net.echo.brain4j.convolution.Kernel;
import net.echo.brain4j.convolution.impl.ConvLayer;
import net.echo.brain4j.layer.Layer;

public class StatesCache {

    private final double[] gradients;
    private final double[] valuesCache;
    private final double[] deltasCache;

    public StatesCache() {
        this.gradients = new double[Synapse.SYNAPSE_COUNTER];
        this.valuesCache = new double[Neuron.NEURON_COUNTER];
        this.deltasCache = new double[Neuron.NEURON_COUNTER];
    }

    public double[] getGradients() {
        return this.gradients;
    }

    public double getGradient(int index) {
        return this.gradients[index];
    }

    public double getValue(Neuron neuron) {
        return this.valuesCache[neuron.getId()];
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

    public void setGradient(int index, double gradient) {
        this.gradients[index] = gradient;
    }

    public void addDelta(Neuron neuron, double delta) {
        this.deltasCache[neuron.getId()] += delta;
    }

    public void addGradient(int index, double change) {
        this.gradients[index] += change;
    }
}
