package org.brain4j.core.model.impl;

import org.brain4j.core.graph.GraphExporter;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.Node;
import org.brain4j.core.model.Model;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.tensor.Tensor;

import java.util.*;
import java.util.stream.IntStream;

public class Graph implements Model {

    private final List<Node> input;
    private final List<Node> output;
    private final List<Node> topology;
    private final Device device;
    private final int seed;

    protected Graph(List<Node> output, Device device, int seed) {
        this(output, device, seed, false);
    }

    private Graph(List<Node> output, Device device, int seed, boolean isCopy) {
        this.device = device;
        this.output = output;
        this.seed = seed;
        this.input = new ArrayList<>();
        this.topology = new ArrayList<>();

        Set<Node> visited = new HashSet<>();

        for (Node outNode : output) {
            dfs(outNode, visited);
        }

        if (isCopy) {
            return;
        }

        for (Node in : input) {
            in.build();
        }

        for (Node topologyNode : topology) {
            topologyNode.build();
        }

        IntStream.range(0, topology.size()).parallel().forEach(i -> {
            Node topologyNode = topology.get(i);
            topologyNode.initWeights(seed + i);
        });
    }

    public static Graph of(Node... output) {
        return new Graph(List.of(output), null, 42);
    }

    public static Graph of(int seed, Node... output) {
        return new Graph(List.of(output), null, seed);
    }

    public static Graph of(int seed, Device device, Node... output) {
        return new Graph(List.of(output), device, seed);
    }

    private void dfs(Node node, Set<Node> visited) {
        if (visited.contains(node)) return;

        visited.add(node);

        for (Node input : node.inputs()) {
            dfs(input, visited);
        }

        if (node.inputs().isEmpty()) {
            input.add(node);
        } else {
            topology.add(node);
        }
    }

    @Override
    public Tensor[] predict(StatesCache cache, Tensor... inputs) {
        if (input.size() != inputs.length) {
            throw Commons.illegalArgument("DAG expects %s inputs but %s were given!", input.size(), inputs.length);
        }

        Map<Node, Tensor[]> outputs = new HashMap<>();

        for (int i = 0; i < input.size(); i++) {
            Tensor source = inputs[i];
            outputs.put(input.get(i), new Tensor[]{cache.isTraining() ? source.withGrad() : source});
        }

        for (Node node : topology) {
            node.forward(cache, outputs);
        }

        List<Tensor> finalOutputs = new ArrayList<>();

        for (Node outNode : output) {
            Tensor[] nodeOutputs = outputs.get(outNode);
            Collections.addAll(finalOutputs, nodeOutputs);
        }

        return finalOutputs.toArray(new Tensor[0]);
    }

    @Override
    public Device device() {
        return device;
    }

    @Override
    public void summary() {
        throw Commons.illegalState("summary() is not supported in Graph impl. Use export(GraphExporter) instead.");
    }
    
    @Override
    public Graph fork(Device device) {
        Map<Node, Node> cache = new HashMap<>();
        List<Node> copy = output.stream().map(n -> n.copy(cache)).toList();
        // Move all copied layers to the new device
        for (Node n : cache.values()) {
            n.layer().to(device);
        }
        return new Graph(copy, device, seed, true);
    }

    @Override
    public Graph copy() {
        Map<Node, Node> cache = new HashMap<>();
        List<Node> copy = output.stream().map(n -> n.copy(cache)).toList();
        return new Graph(copy, device, seed, true);
    }

    @Override
    public List<Layer> getLayers() {
        return topology.stream().map(Node::layer).toList();
    }
    
    public List<Node> input() {
        return input;
    }
    
    public List<Node> output() {
        return output;
    }
    
    public List<Node> topology() {
        return topology;
    }
    
    public int seed() {
        return seed;
    }
}
