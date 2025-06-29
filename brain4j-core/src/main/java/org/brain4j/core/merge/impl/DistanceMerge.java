package org.brain4j.core.merge.impl;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.core.distance.DistanceOperator;
import org.brain4j.core.merge.MergeStrategy;

public class DistanceMerge implements MergeStrategy {

    private final DistanceOperator distanceOperator;

    public DistanceMerge(DistanceOperator distanceOperator) {
        this.distanceOperator = distanceOperator;
    }

    @Override
    public Tensor process(Tensor... inputs) {
        if (inputs.length < 2) {
            throw new IllegalArgumentException("Merging requires at least two input tensors.");
        }

        return distanceOperator.distance(inputs[0], inputs[1]);
    }
}
