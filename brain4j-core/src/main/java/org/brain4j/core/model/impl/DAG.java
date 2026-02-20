package org.brain4j.core.model.impl;

import org.brain4j.core.layer.Layer0;
import org.brain4j.core.layer.Node;
import org.brain4j.core.model.Model;
import org.brain4j.core.model.ModelSpecs;
import org.brain4j.core.training.wrappers.EvaluationResult;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.gpu.silicon.SiliconDevice;
import org.brain4j.math.loss.LossFunction;
import org.brain4j.math.tensor.Tensor;

import java.util.*;

public class DAG implements Model {
    
    private final List<Node> input;
    private final List<Node> output;
    private final List<Node> topology;
    
    private DAG(List<Node> output, int seed) {
        this.input = new ArrayList<>();
        this.output = output;
        this.topology = new ArrayList<>();
        
        Set<Node> visited = new HashSet<>();
        
        for (Node outNode : output) {
            dfs(outNode, visited);
        }
        
        int current = 0;
        
        for (Node topologyNode : topology) {
            topologyNode.build(seed + (current++));
        }
    }
    
    public static DAG of(Node... output) {
        return new DAG(List.of(output), 42);
    }
    
    private void dfs(Node node, Set<Node> visited) {
        if (visited.contains(node)) return;
        
        visited.add(node);
        
        for (Node input : node.getInputs()) {
            dfs(input, visited);
        }
        
        if (node.getInputs().isEmpty()) {
            input.add(node);
        }
        
        topology.add(node);
    }
    
    @Override
    public Tensor[] predict(StatesCache cache, Tensor... inputs) {
        if (input.size() != inputs.length) {
            throw Commons.illegalArgument("DAG expects %s inputs but %s were given!", input.size(), inputs.length);
        }
        
        Tensor[] out = new Tensor[output.size()];
        Map<Node, Tensor[]> outputs = new HashMap<>();
        
        for (int i = 0; i < input.size(); i++) {
            outputs.put(input.get(i), new Tensor[] { inputs[i] });
        }
        
        for (Node node : topology) {
            out = node.forward(cache, outputs);
        }
        
        // TODO: Multi out node support
        return out;
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
    public Model fork(SiliconDevice device) {
        return null;
    }
    
    @Override
    public void summary() {
    
    }
    
    @Override
    public ModelSpecs getSpecs() {
        return null;
    }
    
    @Override
    public SiliconDevice getDevice() {
        return null;
    }
    
    @Override
    public List<Layer0> getLayers() {
        return List.of();
    }
}
