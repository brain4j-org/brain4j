package org.brain4j.core.importing.format.impl;

import org.brain4j.core.Brain4J;
import org.brain4j.core.importing.format.BinaryFormat;
import org.brain4j.core.importing.onnx.ProtoOnnx.*;
import org.brain4j.core.model.impl.Graph;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.impl.*;
import org.brain4j.math.tensor.Tensor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

public class OnnxFormat implements BinaryFormat<Graph> {
    
    private static final Map<Class<? extends Activation>, String> ACTIVATION_MAP = Map.of(
        ReLU.class, "Relu",
        GELU.class, "Gelu",
        Softmax.class, "Softmax",
        Sigmoid.class, "Sigmoid",
        Tanh.class, "Tanh",
        LeakyReLU.class, "LeakyReLU"
    );
    
    @Override
    public Graph deserialize(File file) {
        try {
            byte[] data = Files.readAllBytes(file.toPath());
            
            ModelProto modelProto = ModelProto.parseFrom(data);
            GraphProto graphProto = modelProto.getGraph();
            
            // TODO
            return null;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
    
    @Override
    public void serialize(Graph model, File file) {
        if (model.device() != null) model = model.fork(null);
        
        GraphProto.Builder graphBuilder = GraphProto.newBuilder();
        
        // TODO
        
        graphBuilder.setName(file.getName());
        OperatorSetIdProto opset = OperatorSetIdProto.newBuilder()
            .setDomain("")
            .setVersion(13)
            .build();
        
        ModelProto modelProto = ModelProto.newBuilder()
            .setIrVersion(9)
            .setProducerName("Brain4J")
            .setProducerVersion(Brain4J.getVersion())
            .setGraph(graphBuilder)
            .addOpsetImport(opset)
            .build();
        
        try (FileOutputStream out = new FileOutputStream(file)) {
            out.write(modelProto.toByteArray());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
    
    private String generateName(AtomicInteger counter, Tensor tensor,
                                Map<Tensor, String> tensorNames, Map<Tensor,String> weightsMap) {
        if (weightsMap.containsKey(tensor)) return weightsMap.get(tensor);
        return tensorNames.computeIfAbsent(tensor, t -> "tensor_" + counter.getAndIncrement());
    }
    
    private void extractInput(Tensor input, GraphProto.Builder graphBuilder, AtomicInteger counter,
                              Map<Tensor, String> tensorNames, Map<Tensor, String> weightsMap) {
        ValueInfoProto.Builder proto = ValueInfoProto.newBuilder();
        extractTensor(input, counter, tensorNames, weightsMap, proto, input.shape());
        graphBuilder.addInput(proto);
    }
    
    private void extractOutput(Tensor output, GraphProto.Builder graphBuilder, AtomicInteger counter,
                               Map<Tensor, String> tensorNames, Map<Tensor, String> weightsMap) {
        ValueInfoProto.Builder proto = ValueInfoProto.newBuilder();
        extractTensor(output, counter, tensorNames, weightsMap, proto, output.shape());
        graphBuilder.addOutput(proto);
    }
    
    private void extractTensor(Tensor tensor, AtomicInteger counter, Map<Tensor, String> tensorNames,
                               Map<Tensor, String> weightsMap, ValueInfoProto.Builder proto, int[] shape) {
        TypeProto.Tensor.Builder tensorProto = TypeProto.Tensor.newBuilder();
        TensorShapeProto.Builder shapeProto = TensorShapeProto.newBuilder();
        
        for (int dim : shape) shapeProto.addDim(TensorShapeProto.Dimension.newBuilder().setDimValue(dim));
        
        tensorProto.setElemType(TensorProto.DataType.FLOAT.getNumber()).setShape(shapeProto);
        
        proto.setName(generateName(counter, tensor, tensorNames, weightsMap))
            .setType(TypeProto.newBuilder().setTensorType(tensorProto));
    }
    
    private TensorProto serializeTensor(String name, Tensor tensor) {
        TensorProto.Builder builder = TensorProto.newBuilder()
            .setName(name)
            .setDataType(TensorProto.DataType.FLOAT.getNumber());
        
        for (long dim : tensor.shape()) builder.addDims(dim);
        for (float val : tensor.data()) builder.addFloatData(val);
        
        return builder.build();
    }
    
    private Tensor deserializeTensor(TensorProto tensor) {
        List<Float> raw = tensor.getFloatDataList();
        int[] shape = tensor.getDimsList().stream().mapToInt(Long::intValue).toArray();
        
        Tensor result = Tensors.create(shape);
        float[] data = result.data();
        
        for (int i = 0; i < data.length; i++) data[i] = raw.get(i);
        
        return result;
    }
}

