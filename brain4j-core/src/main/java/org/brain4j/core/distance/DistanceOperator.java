package org.brain4j.core.distance;

import org.brain4j.common.tensor.Tensor;

public interface DistanceOperator {

    Tensor distance(Tensor a, Tensor b);
}
