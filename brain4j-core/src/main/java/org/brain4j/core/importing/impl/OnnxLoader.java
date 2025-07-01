package org.brain4j.core.importing.impl;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.common.tensor.Tensors;
import org.brain4j.common.tensor.autograd.Operation;
import org.brain4j.core.importing.ModelLoader;
import org.brain4j.core.importing.onnx.Onnx;
import org.brain4j.core.model.Model;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class OnnxLoader implements ModelLoader {
    
    @Override
    public Model deserialize(byte[] bytes) throws Exception {
        Onnx.ModelProto proto = Onnx.ModelProto.parseFrom(bytes);
        Onnx.GraphProto graph = proto.getGraph();
        
        Map<String, Tensor> weights = new HashMap<>();
        
        for (Onnx.TensorProto tensor : graph.getInitializerList()) {
            System.out.println("Adding tensor " + tensor.getName());
            weights.put(tensor.getName(), convertTensor(tensor));
        }

        for (Onnx.NodeProto node : graph.getNodeList()) {
            Operation operation = OPERATION_MAP.get(node.getOpType());

            if (operation == null) {
                throw new IllegalArgumentException("Unknown or missing operation type: " + node.getOpType());
            }

            if (node.getInputCount() != operation.requiredInputs()) {
                throw new IllegalArgumentException(
                    "Node " + node.getOpType() + " requires " + node.getInputCount()
                        + " inputs but opereation requires " + operation.requiredInputs()
                );
            }

            List<String> inputList = node.getInputList();

            System.out.println(inputList);

            for (int i = 0; i < inputList.size(); i++) {
                String input = inputList.get(i);
                if (i > 0) {
                    Tensor weightTensor = weights.get(input);

                    System.out.printf("Node %s has input %s with tensor shape %s %n", node.getOpType(), input, Arrays.toString(weightTensor.shape()));
                }
            }
        }

        return null;
    }
    
    private Tensor convertTensor(Onnx.TensorProto tensor) {
        byte[] rawData = tensor.getRawData().toByteArray();

        ByteBuffer dataBuffer = ByteBuffer.wrap(rawData).order(ByteOrder.LITTLE_ENDIAN);
        List<Long> dimensions = tensor.getDimsList();

        int[] shape = new int[dimensions.size()];

        for (int i = 0; i < shape.length; i++) {
            shape[i] = Math.toIntExact(dimensions.get(i));
        }

        Tensor result = Tensors.create(shape);
        float[] data = result.data();
        
        for (int i = 0; i < data.length; i++) {
            data[i] = dataBuffer.getFloat();
        }
        
        return result;
    }
}
