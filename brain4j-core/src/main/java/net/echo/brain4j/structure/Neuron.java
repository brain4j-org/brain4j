package net.echo.brain4j.structure;

import net.echo.brain4j.structure.cache.Parameters;
import net.echo.brain4j.structure.cache.StatesCache;

// TODO: Remove this
public class Neuron {

    private final int id;

    public Neuron() {
        this.id = Parameters.TOTAL_NEURONS++;
    }

    public int getId() {
        return id;
    }

    public float getDelta(StatesCache cacheHolder) {
        return cacheHolder.getDelta(this);
    }

    public double getValue(StatesCache cacheHolder) {
        return cacheHolder.getValue(this);
    }

}
