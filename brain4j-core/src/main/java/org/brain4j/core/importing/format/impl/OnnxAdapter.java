package org.brain4j.core.importing.format.impl;

import org.brain4j.core.Brain4J;
import org.brain4j.core.graphs.GraphModel;
import org.brain4j.core.graphs.GraphNode;
import org.brain4j.core.importing.format.BinaryAdapter;
import org.brain4j.core.importing.onnx.ProtoOnnx.*;
import org.brain4j.core.layer.Layer0;
import org.brain4j.core.layer.impl.utility.InputLayer;
import org.brain4j.core.model.Model;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.impl.*;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.autograd.AutogradContext;
import org.brain4j.math.tensor.autograd.Operation;
import org.brain4j.math.tensor.autograd.impl.ActivationOperation;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

import static org.brain4j.core.importing.Registries.LAYER_REGISTRY;
import static org.brain4j.core.importing.Registries.ONNX_OPERATIONS_REGISTRY;

public class OnnxAdapter implements BinaryAdapter {
    
    private static final Map<Class<? extends Activation>, String> ACTIVATION_MAP = Map.of(
        ReLU.class, "Relu",
        GELU.class, "Gelu",
        Softmax.class, "Softmax",
        Sigmoid.class, "Sigmoid",
        Tanh.class, "Tanh",
        LeakyReLU.class, "LeakyReLU"
    );
    
    @Override
    public GraphModel deserialize(File file) {
        try {
            byte[] data = Files.readAllBytes(file.toPath());
            
            ModelProto modelProto = ModelProto.parseFrom(data);
            GraphProto graphProto = modelProto.getGraph();
            GraphModel.Builder model = GraphModel.builder();
            
            for (TensorProto tensor : graphProto.getInitializerList()) {
                model.initializer(tensor.getName(), deserializeTensor(tensor));
            }
            
            for (NodeProto node : graphProto.getNodeList()) {
                Operation op = ONNX_OPERATIONS_REGISTRY.toInstance(node.getOpType(), node);
                
                if (op == null) {
                    throw Commons.illegalArgument("Unknown operation: %s", node.getOpType());
                }
                
                if (node.getInputCount() != op.requiredInputs()) {
                    throw new IllegalArgumentException("Node " + node.getOpType() + " requires "
                        + node.getInputCount() + " inputs but operation requires " + op.requiredInputs());
                }
                
                model.addNode(new GraphNode(node.getName(), op, node.getInputList(), node.getOutputList()));
            }
            
            List<String> inputs = graphProto.getInputList().stream().map(ValueInfoProto::getName).toList();
            List<String> outputs = graphProto.getOutputList().stream().map(ValueInfoProto::getName).toList();
            
            return model.inputs(inputs).outputs(outputs).compile();
        } catch (Exception e) {
            e.printStackTrace(System.err);
            return null;
        }
    }
    
    @Override
    public void serialize(Model model, File file) {
        if (model.getDevice() != null) model = model.fork(null);
        
        GraphProto.Builder graphBuilder = GraphProto.newBuilder();
        
        Map<Tensor, String> weightsMap = new HashMap<>();
        Map<Tensor, String> tensorNames = new HashMap<>();
        AtomicInteger counter = new AtomicInteger(0);
        
        addInitializers(model, graphBuilder, weightsMap);
        
        Layer0 inputLayer = model.getLayers().getFirst();
        
        if (!(inputLayer instanceof InputLayer wrapped)) {
            throw Commons.illegalArgument("First layer is not an InputLayer instance!");
        }
        
        Tensor input = Tensors.zeros(wrapped.shape()).unsqueeze();
        Tensor output = model.predict(input.withGrad());
        
        extractInput(input, graphBuilder, counter, tensorNames, weightsMap);
        extractOutput(output, graphBuilder, counter, tensorNames, weightsMap);
        
        List<NodeProto> nodes = buildNodesFromTensor(output, counter, tensorNames, weightsMap);
        Collections.reverse(nodes);
        graphBuilder.addAllNode(nodes);
        
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
            e.printStackTrace(System.err);
        }
    }
    
    private void addInitializers(Model model, GraphProto.Builder graphBuilder, Map<Tensor, String> weightsMap) {
        List<Layer0> layers = model.getLayers();
        
        for (int i = 0; i < layers.size(); i++) {
            Layer0 layer = layers.get(i);
            String layerId = LAYER_REGISTRY.fromClass(layer.getClass());
            
            for (Map.Entry<String, Tensor> weight : layer.weightsMap().entrySet()) {
                String name = String.format("%s.%d.%s", layerId, i, weight.getKey());
                
                weightsMap.put(weight.getValue(), name);
                graphBuilder.addInitializer(serializeTensor(name, weight.getValue()));
            }
        }
    }
    
    private List<NodeProto> buildNodesFromTensor(Tensor output, AtomicInteger counter,
                                                 Map<Tensor, String> tensorNames, Map<Tensor, String> weightsMap) {
        Queue<Tensor> queue = new LinkedList<>();
        Set<Tensor> visited = new HashSet<>();
        List<NodeProto> nodes = new ArrayList<>();
        
        queue.add(output);
        
        while (!queue.isEmpty()) {
            Tensor tensor = queue.poll();
            AutogradContext context = tensor.getAutogradContext();
            
            if (context == null || context.operation() == null) continue;
            
            Operation op = context.operation();
            String opType = op instanceof ActivationOperation act ? extractActivation(act) :
                ONNX_OPERATIONS_REGISTRY.fromClass(op.getClass());
            
            if (opType == null) {
                throw Commons.illegalState("No operation for for %s!", op);
            }
            
            NodeProto.Builder node = NodeProto.newBuilder()
                .setName(opType + "_" + Math.abs(UUID.randomUUID().hashCode()))
                .setOpType(opType);

            // TODO: Serialize for squeezing/layernorm, etc
            
            for (Tensor in : context.inputs()) {
                node.addInput(generateName(counter, in, tensorNames, weightsMap));
                
                if (visited.add(in)) queue.add(in);
            }
            
            node.addOutput(generateName(counter, tensor, tensorNames, weightsMap));
            nodes.add(node.build());
        }
        
        return nodes;
    }
    
    private String extractActivation(ActivationOperation op) {
        return ACTIVATION_MAP.getOrDefault(op.activation().getClass(), "unknown");
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

