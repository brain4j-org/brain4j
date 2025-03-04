package net.echo.brain4j.structure;

import net.echo.brain4j.structure.cache.Parameters;
import net.echo.brain4j.structure.cache.StatesCache;

import java.util.ArrayList;
import java.util.List;

public class Neuron {

    private final List<Synapse> synapses = new ArrayList<>();
    private final int id;

    private double bias;
    private double totalDelta;

    public Neuron() {
        this.id = Parameters.TOTAL_NEURONS++;
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

    public double getDelta(StatesCache cacheHolder) {
        return cacheHolder.getDelta(this);
    }

    public void setDelta(StatesCache cacheHolder, double delta) {
        this.totalDelta += delta;

        cacheHolder.addDelta(this, delta);
    }

    public double getValue(StatesCache cacheHolder) {
        return cacheHolder.getValue(this);
    }

    public void setValue(StatesCache cacheHolder, double value) {
        cacheHolder.setValue(this, value);
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public double getHiddenState(StatesCache cache) {
        return cache.getHiddenState(this);
    }
}
