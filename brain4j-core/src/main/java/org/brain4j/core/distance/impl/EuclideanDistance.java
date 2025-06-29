package org.brain4j.core.distance.impl;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.core.distance.DistanceOperator;

public class EuclideanDistance implements DistanceOperator {

    @Override
    public Tensor distance(Tensor a, Tensor b) {
        // shape = [batch_size, dimension]
        Tensor diff = a.minus(b);
        Tensor sqDiff = diff.pow(2);

        // sums along dimension, result shape = [batch_size, 1]
        Tensor sumSq = sqDiff.sum(1, true);
        return sumSq.sqrt();
    }
}
