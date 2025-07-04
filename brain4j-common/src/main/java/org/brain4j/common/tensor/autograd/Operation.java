package org.brain4j.common.tensor.autograd;

import org.brain4j.common.tensor.Tensor;

public interface Operation {

    default int requiredInputs() {
        return 2;
    }

    Tensor compute(Tensor... inputs);

    Tensor[] backward(Tensor gradOutput, Tensor... inputs);
} 