package org.brain4j.core.layer;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.core.training.StatesCache;

public record ForwardContext(
    StatesCache cache,
    Tensor input,
    int index,
    boolean training
) { }
