package org.brain4j.core.weights;

import org.brain4j.core.weights.impl.HeInit;
import org.brain4j.core.weights.impl.LeCunInit;
import org.brain4j.core.weights.impl.NormalInit;
import org.brain4j.core.weights.impl.NormalXavierInit;

public enum WeightInit {

    HE(new HeInit()),
    LECUN(new LeCunInit()),
    NORMAL(new NormalInit()),
    NORMAL_XAVIER(new NormalXavierInit()),
    UNIFORM_XAVIER(new NormalXavierInit());

    private final WeightInitialization function;

    WeightInit(WeightInitialization function) {
        this.function = function;
    }

    public WeightInitialization function() {
        return function;
    }
}
