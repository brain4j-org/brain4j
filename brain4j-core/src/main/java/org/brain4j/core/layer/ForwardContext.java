package org.brain4j.core.layer;

import org.brain4j.core.training.StatesCache;
import org.brain4j.common.tensor.Tensor;

public record ForwardContext(
    StatesCache cache,
    Tensor input,
    int index,
    boolean training
) { }
