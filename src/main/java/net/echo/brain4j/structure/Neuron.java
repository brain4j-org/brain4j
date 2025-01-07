package net.echo.brain4j.structure;

import com.google.gson.annotations.Expose;

import java.util.ArrayList;
import java.util.List;

public class Neuron {

    private final List<Synapse> synapses = new ArrayList<>();
    private final ThreadLocal<Double> localValue = ThreadLocal.withInitial(() -> 0.0);
    private final ThreadLocal<Double> delta = ThreadLocal.withInitial(() -> 0.0);
    private double totalDelta;
    @Expose private double bias;

    public List<Synapse> getSynapses() {
        return synapses;
    }

    public void addSynapse(Synapse synapse) {
        this.synapses.add(synapse);
    }

    public void setTotalDelta(double totalDelta) {
        this.totalDelta = totalDelta;
    }

    public double getTotalDelta() {
        return totalDelta;
    }

    public double getDelta() {
        return delta.get();
    }

    public void setDelta(double delta) {
        this.delta.set(this.delta.get() + delta);
        this.totalDelta += delta;
    }

    public double getValue() {
        return localValue.get();
    }

    public void setValue(double value) {
        this.localValue.set(value);
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }
}
