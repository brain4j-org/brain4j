package org.brain4j.core.importing.format.impl;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.random.RandomGenerator;

import com.google.protobuf.ByteString;

import org.brain4j.core.Brain4J;
import org.brain4j.core.importing.format.BinaryFormat;
import org.brain4j.core.importing.io.OnnxIO;
import org.brain4j.core.importing.onnx.ProtoOnnx.GraphProto;
import org.brain4j.core.importing.onnx.ProtoOnnx.ModelProto;
import org.brain4j.core.importing.onnx.ProtoOnnx.NodeProto;
import org.brain4j.core.importing.onnx.ProtoOnnx.OperatorSetIdProto;
import org.brain4j.core.importing.onnx.ProtoOnnx.TensorProto;
import org.brain4j.core.importing.onnx.ProtoOnnx.TensorShapeProto;
import org.brain4j.core.importing.onnx.ProtoOnnx.TypeProto;
import org.brain4j.core.importing.onnx.ProtoOnnx.ValueInfoProto;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.Node;
import org.brain4j.core.model.impl.Graph;
import org.brain4j.math.Tensors;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.operations.AutogradContext;
import org.brain4j.math.operations.Operation;

/**
 * Imports and exports graph models in ONNX format.
 * Use it to exchange models with external frameworks.
 */
public class OnnxFormat implements BinaryFormat<Graph> {

    /**
     * {@inheritDoc}
     */
    @Override
    public Graph deserialize(File file) {
        try {
            byte[] data = Files.readAllBytes(file.toPath());
            ModelProto modelProto = ModelProto.parseFrom(data);
            GraphProto graphProto = modelProto.getGraph();

            Map<String, Tensor> initializerMap = new HashMap<>();

            for (TensorProto tensorProto : graphProto.getInitializerList()) {
                Tensor tensor = deserializeTensor(tensorProto);
                initializerMap.put(tensorProto.getName(), tensor);
            }

            Map<String, Node> tensorToNode = new HashMap<>();

            for (ValueInfoProto inputInfo : graphProto.getInputList()) {
                String name = inputInfo.getName();

                if (initializerMap.containsKey(name)) {
                    continue;
                }

                TypeProto.Tensor tensorType = inputInfo.getType().getTensorType();
                TensorShapeProto shapeProto = tensorType.getShape();

                int[] dims = shapeProto.getDimList().stream()
                    .mapToInt(d -> (int) d.getDimValue())
                    .toArray();

                int[] layerShape;

                if (dims.length > 1 && dims[0] == 1) {
                    layerShape = Arrays.copyOfRange(dims, 1, dims.length);
                } else if (dims.length == 1 && dims[0] == 1) {
                    layerShape = new int[]{1};
                } else {
                    layerShape = dims;
                }

                if (layerShape.length == 0) {
                    layerShape = new int[]{1};
                }

                Node inputNode = Node.input(Shape.of(layerShape));
                tensorToNode.put(name, inputNode);
            }

            // Retry nodes whose inputs are not resolved yet, since ONNX order is not guaranteed topological.
            List<NodeProto> pending = new ArrayList<>(graphProto.getNodeList());

            while (!pending.isEmpty()) {
                List<NodeProto> deferred = new ArrayList<>();
                boolean progressed = false;

                for (NodeProto nodeProto : pending) {
                    String opType = nodeProto.getOpType();
                    Operation op = OnnxIO.decode(nodeProto);

                    if (op == null) {
                        throw Commons.illegalArgument("Unknown ONNX operation: %s", opType);
                    }

                    List<String> inputNames = nodeProto.getInputList();
                    List<String> outputNames = nodeProto.getOutputList();

                    Map<String, Tensor> nodeConstants = new HashMap<>();
                    List<Node> nodeInputs = new ArrayList<>();
                    boolean resolvable = true;

                    for (String inName : inputNames) {
                        Tensor constTensor = initializerMap.get(inName);

                        if (constTensor != null) {
                            nodeConstants.put(inName, constTensor);
                        } else {
                            Node producer = tensorToNode.get(inName);

                            if (producer == null) {
                                resolvable = false;
                                break;
                            }

                            nodeInputs.add(producer);
                        }
                    }

                    if (!resolvable) {
                        deferred.add(nodeProto);
                        continue;
                    }

                    progressed = true;

                    Layer layer = new OnnxOperationLayer(op, inputNames, nodeConstants);
                    Node node = layer.apply(nodeInputs.toArray(Node[]::new));

                    for (String outName : outputNames) {
                        tensorToNode.put(outName, node);
                    }
                }

                if (!progressed && !deferred.isEmpty()) {
                    throw Commons.illegalState("Unresolvable node cycle for %d nodes (e.g. '%s')",
                        deferred.size(), deferred.getFirst().getName());
                }

                pending = deferred;
            }

            List<Node> outputNodes = new ArrayList<>();

            for (ValueInfoProto outInfo : graphProto.getOutputList()) {
                String outName = outInfo.getName();
                Node outNode = tensorToNode.get(outName);

                if (outNode == null) {
                    throw Commons.illegalState("Output tensor '%s' has no producer", outName);
                }

                outputNodes.add(outNode);
            }

            if (outputNodes.isEmpty()) {
                throw Commons.illegalState("ONNX graph has no outputs");
            }

            return Graph.of(outputNodes.toArray(Node[]::new));
        } catch (Exception e) {
            throw new RuntimeException("Failed to deserialize ONNX model", e);
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void serialize(Graph model, File file) {
        if (model.device() != null) {
            model = model.fork(null);
        }

        GraphProto.Builder graphBuilder = GraphProto.newBuilder();

        Map<Tensor, String> weightsMap = Collections.synchronizedMap(new IdentityHashMap<>());
        Map<Tensor, String> tensorNames = Collections.synchronizedMap(new IdentityHashMap<>());
        AtomicInteger counter = new AtomicInteger(0);

        addInitializers(model, graphBuilder, weightsMap);

        List<Node> inputNodes = model.input();

        if (inputNodes.isEmpty()) {
            throw Commons.illegalState("Graph has no input nodes, cannot export to ONNX");
        }

        List<Tensor> dummyInputsWithGrad = new ArrayList<>();

        for (Node inputNode : inputNodes) {
            List<Shape> inputShapes = inputNode.outputShapes();

            if (inputShapes == null || inputShapes.isEmpty()) {
                throw Commons.illegalState("Input node has no output shapes, cannot export to ONNX");
            }

            Shape shape = inputShapes.getFirst();
            Tensor dummy = Tensors.zeros(shape).unsqueeze();
            dummyInputsWithGrad.add(dummy.withGrad());
        }

        // Run a dummy forward pass to capture the autograd graph for export.
        Tensor[] outputs = model.predict(new StatesCache(true), dummyInputsWithGrad.toArray(Tensor[]::new));

        for (Tensor in : dummyInputsWithGrad) {
            extractInput(in, graphBuilder, counter, tensorNames, weightsMap);
        }

        for (Tensor out : outputs) {
            extractOutput(out, graphBuilder, counter, tensorNames, weightsMap);
        }

        Set<Tensor> graphInputs = Collections.newSetFromMap(new IdentityHashMap<>());
        graphInputs.addAll(dummyInputsWithGrad);

        List<NodeProto> nodes = buildNodesFromTensors(outputs, counter, tensorNames, weightsMap, graphBuilder, graphInputs);
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
            .setProducerVersion(Brain4J.version())
            .setGraph(graphBuilder)
            .addOpsetImport(opset)
            .build();

        try (FileOutputStream out = new FileOutputStream(file)) {
            out.write(modelProto.toByteArray());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private void addInitializers(Graph model, GraphProto.Builder graphBuilder, Map<Tensor, String> weightsMap) {
        List<Layer> layers = model.layers();

        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);

            for (Map.Entry<String, Tensor> entry : layer.parameters().entrySet()) {
                String paramName = entry.getKey();
                Tensor tensor = entry.getValue();
                String name = "layer_" + i + "_" + paramName;

                if (!weightsMap.containsKey(tensor)) {
                    weightsMap.put(tensor, name);
                    graphBuilder.addInitializer(serializeTensor(name, tensor));
                }
            }
        }
    }

    private List<NodeProto> buildNodesFromTensors(Tensor[] outputs, AtomicInteger counter,
                                                  Map<Tensor, String> tensorNames, Map<Tensor, String> weightsMap,
                                                  GraphProto.Builder graphBuilder, Set<Tensor> graphInputs) {
        Queue<Tensor> queue = new LinkedList<>(Arrays.asList(outputs));
        Set<Tensor> visited = Collections.newSetFromMap(new IdentityHashMap<>());
        List<NodeProto> nodes = new ArrayList<>();

        visited.addAll(Arrays.asList(outputs));

        while (!queue.isEmpty()) {
            Tensor tensor = queue.poll();
            AutogradContext ctx = tensor.autogradContext();

            if (ctx == null || ctx.operation() == null) {
                continue;
            }

            Operation op = ctx.operation();
            String opType = OnnxIO.encodeType(op);

            if (opType == null) {
                throw Commons.illegalState("No ONNX mapping for operation %s", op.getClass().getName());
            }

            NodeProto.Builder nodeBuilder = NodeProto.newBuilder()
                .setName(opType + "_" + Math.abs(UUID.randomUUID().hashCode()))
                .setOpType(opType);

            OnnxIO.encode(op, nodeBuilder);

            for (Tensor in : ctx.inputs()) {
                String inputName = generateName(counter, in, tensorNames, weightsMap);
                nodeBuilder.addInput(inputName);

                if (!visited.add(in)) {
                    continue;
                }

                AutogradContext inputCtx = in.autogradContext();

                if (inputCtx != null && inputCtx.operation() != null) {
                    queue.add(in);
                } else if (!graphInputs.contains(in) && !weightsMap.containsKey(in)) {
                    // Emit leaf constants as initializers so import can resolve them without a producer.
                    weightsMap.put(in, inputName);
                    graphBuilder.addInitializer(serializeTensor(inputName, in));
                }
            }

            nodeBuilder.addOutput(generateName(counter, tensor, tensorNames, weightsMap));
            nodes.add(nodeBuilder.build());
        }

        return nodes;
    }

    private String generateName(AtomicInteger counter, Tensor tensor,
                                Map<Tensor, String> tensorNames, Map<Tensor, String> weightsMap) {
        String w = weightsMap.get(tensor);

        return w != null ? w : tensorNames.computeIfAbsent(tensor, _ -> "tensor_" + counter.getAndIncrement());
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

        for (int dim : shape) {
            shapeProto.addDim(TensorShapeProto.Dimension.newBuilder().setDimValue(dim));
        }

        tensorProto.setElemType(TensorProto.DataType.FLOAT.getNumber()).setShape(shapeProto);
        proto.setName(generateName(counter, tensor, tensorNames, weightsMap))
            .setType(TypeProto.newBuilder().setTensorType(tensorProto));
    }

    private TensorProto serializeTensor(String name, Tensor tensor) {
        TensorProto.Builder builder = TensorProto.newBuilder()
            .setName(name)
            .setDataType(TensorProto.DataType.FLOAT.getNumber());

        for (long dim : tensor.shape()) {
            builder.addDims(dim);
        }

        for (float val : tensor.data()) {
            builder.addFloatData(val);
        }

        return builder.build();
    }

    private Tensor deserializeTensor(TensorProto tensor) {
        int[] shape = tensor.getDimsList().stream().mapToInt(Long::intValue).toArray();
        Tensor result;
        ByteString raw = tensor.getRawData();

        if (!raw.isEmpty()) {
            byte[] bytes = raw.toByteArray();
            float[] data = new float[bytes.length / 4];

            ByteBuffer buffer = ByteBuffer.wrap(bytes)
                .order(ByteOrder.LITTLE_ENDIAN);

            for (int i = 0; i < data.length; i++) {
                data[i] = buffer.getFloat();
            }

            result = Tensors.create(shape, data);
        } else {
            List<Float> rawList = tensor.getFloatDataList();
            result = Tensors.create(shape);

            float[] data = result.data();
            int len = Math.min(data.length, rawList.size());

            for (int i = 0; i < len; i++) {
                data[i] = rawList.get(i);
            }
        }

        return result;
    }

    /**
     * Wraps a single ONNX operation as a graph layer.
     * Use it to import ONNX nodes that have no direct layer mapping.
     */
    public static class OnnxOperationLayer extends Layer {

        private final Operation op;
        private final List<String> inputNames;
        private final Map<String, Tensor> constants;

        /**
         * Create a layer for the given operation and wiring.
         */
        public OnnxOperationLayer(Operation op, List<String> inputNames, Map<String, Tensor> constants) {
            this.op = op;
            this.inputNames = List.copyOf(inputNames);
            this.constants = new HashMap<>(constants);
        }

        /**
         * Return the wrapped operation.
         *
         * @return the operation
         */
        public Operation operation() {
            return op;
        }

        /**
         * Return the ONNX input names in order.
         *
         * @return the input names
         */
        public List<String> inputNames() {
            return inputNames;
        }

        /**
         * Return constant tensors keyed by input name.
         *
         * @return the constants
         */
        public Map<String, Tensor> constants() {
            return constants;
        }

        /**
         * {@inheritDoc}
         */
        @Override
        public void build(List<Shape> inputShapes) {
        }

        /**
         * {@inheritDoc}
         */
        @Override
        public void initWeights(List<Shape> inputShapes, RandomGenerator rng) {
        }

        /**
         * {@inheritDoc}
         */
        @Override
        public List<Shape> inferOutputShapes(List<Shape> inputShapes) {
            List<Tensor> dummies = new ArrayList<>();
            int idx = 0;

            for (String name : inputNames) {
                Tensor c = constants.get(name);

                if (c != null) {
                    dummies.add(Tensors.zeros(Shape.of(c.shape())));
                } else {
                    if (idx >= inputShapes.size()) {
                        throw Commons.illegalState("Not enough input shapes for ONNX op %s: expected at least %d non-constant inputs",
                            op.getClass().getSimpleName(), inputNames.size() - constants.size());
                    }

                    Shape s = inputShapes.get(idx++);
                    // Prepend batch 1 since runtime tensors carry a batch dim while static shapes do not.
                    int[] dims = s.dims();
                    int[] withBatch = new int[dims.length + 1];
                    withBatch[0] = 1;
                    System.arraycopy(dims, 0, withBatch, 1, dims.length);
                    dummies.add(Tensors.zeros(Shape.of(withBatch)));
                }
            }

            Tensor out = op.compute(dummies.toArray(Tensor[]::new));
            // Strip the batch dim again to keep the graph convention without batch.
            int[] outShape = out.shape();

            if (outShape.length > 0 && outShape[0] == 1) {
                outShape = Arrays.copyOfRange(outShape, 1, outShape.length);

                if (outShape.length == 0) {
                    outShape = new int[]{1};
                }
            }

            return List.of(Shape.of(outShape));
        }

        /**
         * {@inheritDoc}
         */
        @Override
        public Tensor[] forward(StatesCache cache, Tensor... inputs) {
            Tensor[] full = new Tensor[inputNames.size()];
            int idx = 0;

            for (int i = 0; i < inputNames.size(); i++) {
                String name = inputNames.get(i);
                Tensor c = constants.get(name);

                if (c != null) {
                    full[i] = c;
                } else {
                    if (idx >= inputs.length) {
                        throw Commons.illegalState("Missing input for ONNX op %s at position %d (name=%s)",
                            op.getClass().getSimpleName(), i, name);
                    }

                    full[i] = inputs[idx++];
                }
            }

            return new Tensor[]{ op.compute(full) };
        }

        /**
         * {@inheritDoc}
         */
        @Override
        public Layer copy() {
            return new OnnxOperationLayer(op, inputNames, constants);
        }
    }
}
