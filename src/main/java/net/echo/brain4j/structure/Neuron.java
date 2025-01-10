package net.echo.brain4j.structure;

import com.google.gson.annotations.Expose;
import net.echo.brain4j.threading.NeuronCacheHolder;

import java.util.ArrayList;
import java.util.List;

public class Neuron {

    public static int NEURON_COUNTER = 0;

    private final List<Synapse> synapses = new ArrayList<>();
    private final ThreadLocal<Double> localValue = ThreadLocal.withInitial(() -> 0.0);
    private final ThreadLocal<Double> delta = ThreadLocal.withInitial(() -> 0.0);
    private final int id;

    @Expose
    private double bias;
    private double totalDelta;

    public Neuron() {
        this.id = NEURON_COUNTER++;
    }

    public List<Synapse> getSynapses() {
        return synapses;
    }

    public void addSynapse(Synapse synapse) {
        synapses.add(synapse);
    }

    public void setTotalDelta(double totalDelta) {
        this.totalDelta = totalDelta;
    }

    public double getTotalDelta() {
        return totalDelta;
    }

    public int getId() {
        return id;
    }

    public double getDelta(NeuronCacheHolder cacheHolder) {
        if (cacheHolder == null) {
            return delta.get();
        }

        return cacheHolder.getDelta(this);
    }

    public void setDelta(NeuronCacheHolder cacheHolder, double delta) {
        this.totalDelta += delta;

        if (cacheHolder == null) {
            this.delta.set(this.delta.get() + delta);
            return;
        }

        cacheHolder.addDelta(this, delta);
    }

    public double getValue(NeuronCacheHolder cacheHolder) {
        if (cacheHolder == null) {
            return localValue.get();
        }

        return cacheHolder.getValue(this);
    }

    public void setValue(NeuronCacheHolder cacheHolder, double value) {
        if (cacheHolder == null) {
            localValue.set(value);
            return;
        }

        cacheHolder.setValue(this, value);
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }
}
