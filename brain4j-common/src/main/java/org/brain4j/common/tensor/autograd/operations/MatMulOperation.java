package org.brain4j.common.tensor.autograd.operations;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.common.tensor.autograd.Operation;

public class MatMulOperation implements Operation {

    @Override
    public Tensor forward(Tensor... inputs) {
        return inputs[0].matmul(inputs[1]);
    }
    
    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor... inputs) {
        Tensor a = inputs[0];
        Tensor b = inputs[1];
        
        // For matrix multiplication: C = A @ B
        // dL/dA = dL/dC @ B.T
        Tensor gradA = gradOutput.matmul(b.transpose());
        
        // dL/dB = A.T @ dL/dC
        Tensor gradB = a.transpose().matmul(gradOutput);

        return new Tensor[] { gradA, gradB };
    }
} 