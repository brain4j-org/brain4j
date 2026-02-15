package org.brain4j.math.tensor.autograd.impl;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.autograd.Operation;

public class MatMulOperation implements Operation {
    
    @Override
    public Tensor compute(Tensor... inputs) {
        return inputs[0].matmul(inputs[1]);
    }
    
    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor output, Tensor... inputs) {
        Tensor a = inputs[0];
        Tensor b = inputs[1];
        
        Tensor aT = a.transpose();
        Tensor bT = b.transpose();
        
        Tensor gradA = gradOutput.matmul(bT);
        Tensor gradB = aT.matmul(gradOutput);
        
        return new Tensor[] { gradA, gradB };
    }
} 