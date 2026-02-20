package org.brain4j.math.data;

import org.brain4j.math.tensor.Tensor;

import java.util.HashMap;
import java.util.Map;

public class StatesCache {

    private final Map<Object, Tensor> tensorCache;
    private final Map<String, Tensor[]> states;
    
    private final boolean training;

    public static StatesCache withTraining() {
        return new StatesCache(true);
    }

    public StatesCache() {
        this(false);
    }

    public StatesCache(boolean training) {
        this.training = training;
        this.states = new HashMap<>();
        this.tensorCache = new HashMap<>();
    }

    public boolean isTraining() {
        return training;
    }

    public Tensor get(Object key) {
        return tensorCache.get(key);
    }

    public void set(Object key, Tensor value) {
        tensorCache.put(key, value);
    }
    
    public Tensor[] getStates(Object key, String id) {
        return states.get(key.hashCode() + id);
    }
    
    public void setStates(Object key, String id, Tensor... values) {
        states.put(key.hashCode() + id, values);
    }
}

