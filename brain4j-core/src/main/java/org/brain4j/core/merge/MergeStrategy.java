package org.brain4j.core.merge;

import org.brain4j.math.tensor.Tensor;

public interface MergeStrategy {

    Tensor process(Tensor... inputs);

    Tensor inverse(int[] dimensions, Tensor input);
}
