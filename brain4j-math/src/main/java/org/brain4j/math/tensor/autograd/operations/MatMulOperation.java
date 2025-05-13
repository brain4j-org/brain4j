package org.brain4j.math.tensor.autograd.operations;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.autograd.Operation;

import java.util.Arrays;

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

        System.out.println("Matmul op");
        System.out.println(Arrays.toString(gradOutput.shape()));
        System.out.println(Arrays.toString(b.transpose().shape()));
        Tensor gradA = gradOutput.matmul(b.transpose());
        
        // dL/dB = A.T @ dL/dC
        Tensor gradB = a.transpose().matmul(gradOutput);
        
        return new Tensor[] { gradA, gradB };
    }
} 