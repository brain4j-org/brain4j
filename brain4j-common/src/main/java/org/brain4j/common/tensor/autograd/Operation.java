package org.brain4j.common.tensor.autograd;

import org.brain4j.common.tensor.Tensor;

public interface Operation {

    Tensor forward(Tensor... inputs);

    Tensor[] backward(Tensor gradOutput, Tensor... inputs);
} 