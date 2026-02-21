package org.brain4j.core.model.impl;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.Node;
import org.brain4j.core.model.Model;
import org.brain4j.core.training.wrappers.EvaluationResult;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.gpu.silicon.SiliconDevice;
import org.brain4j.math.loss.LossFunction;
import org.brain4j.math.tensor.Tensor;

import java.util.*;
import java.util.stream.IntStream;

public class DAG implements Model {
    
    private final List<Node> input;
    private final List<Node> output;
    private final List<Node> topology;
    private final SiliconDevice device;
    private final int seed;
    
    protected DAG(List<Node> output, SiliconDevice device, int seed) {
        this.device = device;
        this.output = output;
        this.seed = seed;
        this.input = new ArrayList<>();
        this.topology = new ArrayList<>();

        Set<Node> visited = new HashSet<>();
        
        for (Node outNode : output) {
            dfs(outNode, visited);
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
    
    public static DAG of(Node... output) {
        return new DAG(List.of(output), null, 42);
    }
    
    public static DAG of(int seed, Node... output) {
        return new DAG(List.of(output), null, seed);
    }
    
    public static DAG of(int seed, SiliconDevice device, Node... output) {
        return new DAG(List.of(output), device, seed);
    }
    
    private void dfs(Node node, Set<Node> visited) {
        if (visited.contains(node)) return;
        
        visited.add(node);
        
        for (Node input : node.getInputs()) {
            dfs(input, visited);
        }
        
        if (node.getInputs().isEmpty()) {
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
            outputs.put(input.get(i), new Tensor[] { inputs[i] });
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
    public SiliconDevice getDevice() {
        return device;
    }
    
    @Override
    public EvaluationResult evaluate(ListDataSource dataSource, LossFunction lossFunction) {
        return null;
    }
    
    @Override
    public double loss(ListDataSource dataSource, LossFunction lossFunction) {
        return 0;
    }
    
    @Override
    public void summary() {
    
    }
    
    @Override
    public DAG fork(SiliconDevice device) {
        List<Node> copy = output.stream().map(Node::copy).toList();
        return new DAG(copy, device, seed);
    }
    
    @Override
    public DAG copy() {
        List<Node> copy = output.stream().map(Node::copy).toList();
        return new DAG(copy, device, seed);
    }
    
    @Override
    public List<Layer> getLayers() {
        return topology.stream().map(Node::getLayer).toList();
    }
    
    public int seed() {
        return seed;
    }
}
