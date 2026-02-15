package org.brain4j.math.tensor.autograd;

import org.brain4j.math.tensor.Tensor;

public interface Operation {

    default int requiredInputs() {
        return 2;
    }

    Tensor compute(Tensor... inputs);

    Tensor[] backward(Tensor gradOutput, Tensor output, Tensor... inputs);
} 