package org.brain4j.core.graphs;

import org.brain4j.common.data.ListDataSource;
import org.brain4j.common.device.Device;
import org.brain4j.common.tensor.Tensor;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.loss.LossFunction;
import org.brain4j.core.model.Model;
import org.brain4j.core.training.StatesCache;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.core.training.wrappers.EvaluationResult;

import java.util.*;

public class GraphModel implements Model {

    private final List<GraphNode> nodes;
    private final List<String> inputNames;
    private final List<String> outputNames;
    private final Map<String, Tensor> initializers;

    private Device device;

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

    public static Builder newGraph() {
        return new Builder();
    }

    @Override
    public Tensor predict(StatesCache cache, boolean training, Tensor... inputs) {
        if (inputs.length != inputNames.size()) {
            throw new IllegalArgumentException("Expected " + inputNames.size() + " inputs, but got " + inputs.length);
        }

        Map<String, Tensor> computed = new HashMap<>();

        for (int i = 0; i < inputs.length; i++) {
            computed.put(inputNames.get(i), inputs[i]);
        }

        computed.putAll(initializers);

        for (GraphNode node : nodes) {
            List<String> inputNames = node.inputs();
            Tensor[] inputTensors = new Tensor[inputNames.size()];

            for (int j = 0; j < inputTensors.length; j++) {
                Tensor input = computed.get(inputNames.get(j));

                if (input == null) {
                    throw new IllegalStateException("Missing tensor for input: " + inputNames.get(j));
                }

                inputTensors[j] = input;
            }

            Tensor output = node.operation().compute(inputTensors);

            for (String outputName : node.outputs()) {
                computed.put(outputName, output);
            }
        }

        if (outputNames.size() != 1) {
            throw new UnsupportedOperationException("Only single-output graphs are supported in predict()");
        }

        return computed.get(outputNames.getFirst());
    }

    @Override
    public Model add(Layer layer) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Model add(int index, Layer layer) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void backpropagate(StatesCache cache, Tensor outputs, Tensor targets) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void fit(ListDataSource train, ListDataSource validation, int epoches, int evaluateEvery) {
        throw new UnsupportedOperationException();
    }

    @Override
    public EvaluationResult evaluate(ListDataSource dataSource) {
        return null;
    }

    @Override
    public double loss(ListDataSource dataSource) {
        return 0;
    }

    @Override
    public Model compile(LossFunction lossFunction, Optimizer optimizer, Updater updater) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Model to(Device device) {
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
    public Device device() {
        return device;
    }

    @Override
    public List<Layer> layers() {
        return List.of();
    }

    @Override
    public List<Layer> flattened() {
        return List.of();
    }

    @Override
    public Layer layerAt(int index) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Layer flattenedAt(int index) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void summary() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void zeroGrad() {
        throw new UnsupportedOperationException();
    }

    @Override
    public Iterator<Layer> iterator() {
        return null;
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

        public Builder addInitializer(String name, Tensor tensor) {
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
