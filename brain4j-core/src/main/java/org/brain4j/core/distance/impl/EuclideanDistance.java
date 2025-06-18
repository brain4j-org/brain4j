package org.brain4j.core.distance.impl;

import org.brain4j.core.distance.DistanceOperator;
import org.brain4j.math.tensor.Tensor;

public class EuclideanDistance implements DistanceOperator {

    @Override
    public Tensor distance(Tensor a, Tensor b) {
        Tensor diff = a.minus(b);
        Tensor sqDiff = diff.pow(2);

        Tensor sumSq = sqDiff.sum(1, true);
        return sumSq.sqrt();
    }
}
