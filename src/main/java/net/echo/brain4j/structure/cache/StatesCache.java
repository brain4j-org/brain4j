package net.echo.brain4j.structure.cache;

import net.echo.brain4j.convolution.Kernel;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.Neuron;

import java.util.HashMap;
import java.util.Map;

public class StatesCache {

    private final Map<Layer, Kernel> inputKernelCache;
    private final Map<Layer, Kernel> outputKernelCache;
    private final Map<Layer, Kernel> deltaKernelCache;

    private final double[] valuesCache;
    private final double[] deltasCache;
    private double[] previousTimestep;

    public StatesCache() {
        this.valuesCache = new double[Parameters.TOTAL_NEURONS];
        this.deltasCache = new double[Parameters.TOTAL_NEURONS];
        this.inputKernelCache = new HashMap<>();
        this.outputKernelCache = new HashMap<>();
        this.deltaKernelCache = new HashMap<>();
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

    public Kernel getInputKernel(Layer layer) {
        return inputKernelCache.get(layer);
    }

    public void setInputKernel(Layer layer, Kernel kernel) {
        inputKernelCache.put(layer, kernel);
    }

    public Kernel getOutputKernel(Layer layer) {
        return outputKernelCache.get(layer);
    }

    public void setOutputKernel(Layer layer, Kernel kernel) {
        outputKernelCache.put(layer, kernel);
    }

    public Kernel getDeltaKernel(Layer layer) {
        return deltaKernelCache.get(layer);
    }

    public void setDeltaKernel(Layer layer, Kernel kernel) {
        deltaKernelCache.put(layer, kernel);
    }

    public void ensureRecurrentCache() {
        if (previousTimestep != null) return;

        previousTimestep = new double[Parameters.TOTAL_NEURONS];
    }

    public double getHiddenState(Neuron neuron) {
        return previousTimestep[neuron.getId()];
    }

    public void setHiddenState(Neuron neuron, double activatedValue) {
        previousTimestep[neuron.getId()] = activatedValue;
    }
}
