package org.brain4j.math.tensor.autograd;

import org.brain4j.math.tensor.Tensor;

public interface Operation {

    Tensor forward(Tensor... inputs);

    Tensor[] backward(Tensor gradOutput, Tensor... inputs);
} 