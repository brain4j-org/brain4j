package net.echo.brain4j.structure;

import net.echo.brain4j.structure.cache.Parameters;
import net.echo.brain4j.structure.cache.StatesCache;

import java.util.ArrayList;
import java.util.List;

public class Neuron {

    private final int id;

    private float bias;
    private float totalDelta;

    public Neuron() {
        this.id = Parameters.TOTAL_NEURONS++;
    }

    public void setTotalDelta(float totalDelta) {
        this.totalDelta = totalDelta;
    }

    public double getTotalDelta() {
        return totalDelta;
    }

    public int getId() {
        return id;
    }

    public float getDelta(StatesCache cacheHolder) {
        return cacheHolder.getDelta(this);
    }

    public void setDelta(StatesCache cacheHolder, float delta) {
        this.totalDelta += delta;

        cacheHolder.addDelta(this, delta);
    }

    public double getValue(StatesCache cacheHolder) {
        return cacheHolder.getValue(this);
    }

    public void setValue(StatesCache cacheHolder, double value) {
        cacheHolder.setValue(this, (float) value);
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = (float) bias;
    }
}
