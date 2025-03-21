package net.echo.math4j.math.tensor.autograd.operations;

import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorCPU;
import net.echo.math4j.math.tensor.autograd.Operation;

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