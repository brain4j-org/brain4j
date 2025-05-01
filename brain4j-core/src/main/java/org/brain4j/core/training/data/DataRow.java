package org.brain4j.core.training.data;

import org.brain4j.math.tensor.Tensor;

@Deprecated(since = "2.7.0", forRemoval = true)
public record DataRow(Tensor inputs, Tensor outputs) {
}
