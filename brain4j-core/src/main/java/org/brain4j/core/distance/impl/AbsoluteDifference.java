package org.brain4j.core.distance.impl;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.core.distance.DistanceOperator;

public class AbsoluteDifference implements DistanceOperator {

    @Override
    public Tensor distance(Tensor a, Tensor b) {
        return a.sub(b).map(Math::abs);
    }
}
