package org.brain4j.common.tensor.autograd.operations;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.common.tensor.autograd.Operation;

public class ConvolveOperation implements Operation {

    @Override
    public Tensor forward(Tensor... inputs) {
        return inputs[0].convolve(inputs[1], 1, 0);
    }

    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor... inputs) {
        Tensor A = inputs[0];
        Tensor B = inputs[1];

        // dL/dA = dL/dC * flip(B)
        Tensor gradA = gradOutput.convolve(B.flip(), 1, B.shape()[2] - 1);

        // dL/dB = flip(A) * dL/dC
        Tensor gradB = A.flip().convolve(gradOutput, 1, 0);

        return new Tensor[] { gradA, gradB };
    }
}
