package org.brain4j.core.graphs;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.loss.LossFunction;
import org.brain4j.core.model.Model;
import org.brain4j.core.model.ModelSpecs;
import org.brain4j.core.training.wrappers.EvaluationResult;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.gpu.silicon.SiliconDevice;
import org.brain4j.math.tensor.Tensor;

import java.util.*;

/**
 * A neural network model represented as a directed acyclic graph (DAG).
 *
 * <p>Graph models allow for more complex network architectures than sequential
 * models, supporting multiple inputs/outputs and arbitrary connections between
 * nodes. Each node in the graph represents an operation (layer) and edges
 * represent tensor flow between operations.
 *
 * <p>This implementation:
 * <ul>
 *   <li>Supports importing models from frameworks like ONNX
 *   <li>Manages tensor flow between operations
 *   <li>Handles device placement of operations and tensors
 *   <li>Supports inference only (no training)
 * </ul>
 */
public class GraphModel implements Model {

    private final List<GraphNode> nodes;
    private final List<String> inputNames;
    private final List<String> outputNames;
    private final Map<String, Tensor> initializers;

    private SiliconDevice device;

    public GraphModel(
        List<GraphNode> nodes,
        List<String> inputNames,
        List<String> outputNames,
        Map<String, Tensor> initializers
    ) {
        this.nodes = nodes;
        this.inputNames = inputNames;
        this.outputNames = outputNames;
        this.initializers = initializers;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public Tensor[] predict(StatesCache cache, Tensor... inputs) {
        if (inputs.length != inputNames.size()) {
            throw Commons.illegalArgument("Expected %s inputs, but got %s!", inputNames.size(), inputs.length);
        }
        
        if (device != null) {
            device.createResources();
        }

        Map<String, Tensor> computed = new HashMap<>(initializers);

        for (int i = 0; i < inputs.length; i++) {
            computed.put(inputNames.get(i), inputs[i].to(device));
        }

        for (GraphNode node : nodes) {
            List<String> inputNames = node.inputs();
            Tensor[] inputTensors = new Tensor[inputNames.size()];

            for (int j = 0; j < inputTensors.length; j++) {
                Tensor input = computed.get(inputNames.get(j));

                if (input == null) {
                    throw Commons.illegalState("Missing tensor for input: %s for node %s", inputNames.get(j), node.name());
                }

                inputTensors[j] = input.to(device);
            }

            Tensor output = node.operation().compute(inputTensors);

            for (String outputName : node.outputs()) {
                computed.put(outputName, output);
            }
        }

        Tensor[] outputs = new Tensor[outputNames.size()];

        for (int i = 0; i < outputs.length; i++) {
            outputs[i] = computed.get(outputNames.get(i));
        }

        if (device != null && !cache.isTraining()) {
            device.closeResources();
        }

        return outputs;
    }
    
    @Override
    public EvaluationResult evaluate(ListDataSource dataSource, LossFunction lossFunction) {
        return null; // TODO
    }
    
    @Override
    public Model fork(SiliconDevice device) {
        this.device = device;

        Map<String, Tensor> copy = new HashMap<>(initializers);

        initializers.clear();

        for (Map.Entry<String, Tensor> entry : copy.entrySet()) {
            Tensor weight = entry.getValue().to(device);
            initializers.put(entry.getKey(), weight);
        }

        return this;
    }

    @Override
    public SiliconDevice getDevice() {
        return device;
    }
    
    @Override
    public void summary() {
        throw new UnsupportedOperationException();
    }
    
    @Override
    public ModelSpecs getSpecs() {
        throw new UnsupportedOperationException();
    }
    
    @Override
    public List<Layer> getLayers() {
        throw new UnsupportedOperationException();
    }

    public static class Builder {

        private final List<GraphNode> nodes = new ArrayList<>();
        private final Map<String, Tensor> initializers = new HashMap<>();
        private List<String> inputs = new ArrayList<>();
        private List<String> outputs = new ArrayList<>();

        public Builder addNode(GraphNode node) {
            this.nodes.add(node);
            return this;
        }

        public Builder initializer(String name, Tensor tensor) {
            this.initializers.put(name, tensor);
            return this;
        }

        public Builder inputs(List<String> inputs) {
            this.inputs = inputs;
            return this;
        }

        public Builder outputs(List<String> outputs) {
            this.outputs = outputs;
            return this;
        }

        public GraphModel compile() {
            return new GraphModel(nodes, inputs, outputs, initializers);
        }
    }
}
