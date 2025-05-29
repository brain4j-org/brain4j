package org.brain4j.math.tensor.autograd.operations;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.autograd.Operation;

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

//        System.out.println("A:");
//        System.out.println(a.toString("%.2f"));

//        System.out.println("Grad out:");
//        System.out.println(gradOutput.toString("%.4f"));
//
//        System.out.println("Input b:");
//        System.out.println(b.transpose().toString("%.4f"));
//
//        System.out.println("not transposed b:");
//        System.out.println(b.toString("%.4f"));
//
//        System.out.println("Input b hashcode: " + b.hashCode());
//        System.out.println("Gradient of b:");
//        System.out.println(gradB.toString("%.4f"));
//
//        System.out.printf("Gradient of a:");
//        System.out.println(gradA.toString("%.4f"));

        return new Tensor[] { gradA, gradB };
    }
} 