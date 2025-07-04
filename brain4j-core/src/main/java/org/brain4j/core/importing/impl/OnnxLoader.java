package org.brain4j.core.importing.impl;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.common.tensor.Tensors;
import org.brain4j.common.tensor.autograd.Operation;
import org.brain4j.core.graphs.GraphModel;
import org.brain4j.core.graphs.GraphNode;
import org.brain4j.core.importing.ModelLoader;
import org.brain4j.core.importing.onnx.Onnx;
import org.brain4j.core.model.Model;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;

public class OnnxLoader implements ModelLoader {
    
    @Override
    public Model deserialize(byte[] bytes) throws Exception {
        Onnx.ModelProto proto = Onnx.ModelProto.parseFrom(bytes);
        Onnx.GraphProto graph = proto.getGraph();
        
        GraphModel.Builder model = GraphModel.newGraph();

        for (Onnx.TensorProto tensor : graph.getInitializerList()) {
            Tensor weight = convertTensor(tensor);
            model.addInitializer(tensor.getName(), weight);
        }

        for (Onnx.NodeProto node : graph.getNodeList()) {
            Operation operation = OPERATION_MAP.get(node.getOpType());

            if (operation == null) {
                throw new IllegalArgumentException("Unknown or missing operation type: " + node.getOpType());
            }

            if (node.getInputCount() != operation.requiredInputs()) {
                throw new IllegalArgumentException(
                    "Node " + node.getOpType() + " requires " + node.getInputCount()
                        + " inputs but operation requires " + operation.requiredInputs()
                );
            }

            model.addNode(new GraphNode(
                node.getName(),
                operation,
                node.getInputList(),
                node.getOutputList()
            ));
        }


        List<String> inputs = graph.getInputList().stream()
            .map(Onnx.ValueInfoProto::getName)
            .toList();

        List<String> outputs = graph.getOutputList().stream()
            .map(Onnx.ValueInfoProto::getName)
            .toList();

        model.inputs(inputs);
        model.outputs(outputs);

        return model.compile();
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
