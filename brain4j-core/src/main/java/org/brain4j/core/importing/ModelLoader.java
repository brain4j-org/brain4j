package org.brain4j.core.importing;

import org.brain4j.common.tensor.autograd.Operation;
import org.brain4j.common.tensor.autograd.impl.*;
import org.brain4j.core.activation.impl.*;
import org.brain4j.core.model.Model;

import java.util.HashMap;
import java.util.Map;

public interface ModelLoader {

    Map<String, Operation> OPERATION_MAP = new HashMap<>() {
        {
            put("Add", new AddOperation());
            put("Sub", new SubOperation());
            put("Mul", new MulOperation());
            put("Div", new DivOperation());
            put("Gemm", new GemmOperation());
            put("MatMul", new MatMulOperation());

            put("Relu", new ActivationOperation(new ReLUActivation()));
            put("Sigmoid", new ActivationOperation(new SigmoidActivation()));
            put("Tanh", new ActivationOperation(new TanhActivation()));
            put("LeakyRelu", new ActivationOperation(new LeakyReLUActivation()));
            put("Gelu", new ActivationOperation(new GELUActivation()));
            put("Softmax", new ActivationOperation(new SoftmaxActivation()));

            put("LayerNormalization", new LayerNormOperation( 1e-5));
        }
    };
    
    Model deserialize(byte[] bytes) throws Exception;
}
