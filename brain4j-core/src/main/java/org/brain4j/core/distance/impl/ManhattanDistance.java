package org.brain4j.core.distance.impl;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.core.distance.DistanceOperator;

public class ManhattanDistance implements DistanceOperator {

    @Override
    public Tensor distance(Tensor a, Tensor b) {
        Tensor diff = a.minus(b);
        Tensor absDiff = diff.map(Math::abs);

        return absDiff.sum(1, true);
    }
}
