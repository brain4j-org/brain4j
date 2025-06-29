package org.brain4j.core.merge;

import org.brain4j.common.tensor.Tensor;

public interface MergeStrategy {

    Tensor process(Tensor... inputs);
}
