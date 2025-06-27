package org.brain4j.core.importing.impl;

import onnx.Onnx;
import org.brain4j.core.importing.ModelLoader;
import org.brain4j.core.model.Model;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
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
            weights.put(tensor.getName(), convertTensor(tensor));
        }
        
        return null;
    }
    
    private Tensor convertTensor(Onnx.TensorProto tensor) {
        List<Long> dimensions = tensor.getDimsList();
        ByteBuffer buffer = ByteBuffer.wrap(tensor.getRawData().toByteArray()).order(ByteOrder.LITTLE_ENDIAN);
        
        int[] shape = new int[dimensions.size()];
        Tensor result = Tensors.create(shape);
        
        float[] data = result.data();
        
        for (int i = 0; i < data.length; i++) {
            data[i] = buffer.getFloat();
        }
        
        return result;
    }
}
