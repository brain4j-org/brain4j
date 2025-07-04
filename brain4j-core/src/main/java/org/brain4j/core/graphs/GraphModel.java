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
    private final int inputsAmount;

    public GraphModel(int inputsAmount, List<GraphNode> nodes) {
        this.nodes = nodes;
        this.inputsAmount = inputsAmount;
    }

    public static GraphModel.Builder newGraph() {
        return new GraphModel.Builder();
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
    public Tensor predict(StatesCache cache, boolean training, Tensor... inputs) {
        if (inputs.length != this.inputsAmount) {
            throw new IllegalArgumentException(
                "Input array length does not match model's input amount! Got " + inputs.length + " instead of " + this.inputsAmount
            );
        }

        for (GraphNode node : nodes) {

        }

        return null;
    }

    @Override
    public void backpropagate(StatesCache cache, Tensor outputs, Tensor targets) {

    }

    @Override
    public void fit(ListDataSource train, ListDataSource validation, int epoches, int evaluateEvery) {

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
        return null;
    }

    @Override
    public Model to(Device device) {
        return null;
    }

    @Override
    public Device device() {
        return null;
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
        return null;
    }

    @Override
    public Layer flattenedAt(int index) {
        return null;
    }

    @Override
    public void summary() {

    }

    @Override
    public void zeroGrad() {

    }

    @Override
    public Iterator<Layer> iterator() {
        return null;
    }

    public static class Builder {

        private final List<NodeInfo> hidden;
        private int inputs;

        public Builder() {
            this.hidden = new ArrayList<>();
        }

        public Builder input(String name, Layer layer, String output) {
            add(name, layer, output);
            inputs++;
            return this;
        }

        public Builder add(String name, Layer layer, String... outputs) {
            this.hidden.add(new NodeInfo(name, layer, outputs));
            return this;
        }

        public Builder output(String name, Layer layer) {
            this.hidden.add(new NodeInfo(name, layer));
            return this;
        }

        public Builder lossFunction(LossFunction lossFunction) {
            return this;
        }

        public Builder optimizer(Optimizer optimizer) {
            return this;
        }

        public Builder updater(Updater updater) {
            return this;
        }

        public GraphModel compile() {
            if (inputs == 0) {
                throw new IllegalStateException("You must specify at least one input!");
            }


            return new GraphModel(inputs, new ArrayList<>());
        }
    }

    public record NodeInfo(String name, Layer layer, String... outputs) { }
}
