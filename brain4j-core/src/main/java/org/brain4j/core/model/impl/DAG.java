package org.brain4j.core.model.impl;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.Layer0;
import org.brain4j.core.layer.Node;
import org.brain4j.core.model.Model;
import org.brain4j.core.model.ModelSpecs;
import org.brain4j.core.training.wrappers.EvaluationResult;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.gpu.silicon.SiliconDevice;
import org.brain4j.math.loss.LossFunction;
import org.brain4j.math.tensor.Tensor;

import java.util.List;

public class DAG implements Model {
    
    private final List<Node> nodes;
    
    public DAG(List<Node> nodes) {
        this.nodes = nodes;
    }
    
    private void dfs(Node node, List<Node> visited, List<Node> topoOrder) {
        if (visited.contains(node)) return;
        
        visited.add(node);
        
        for (Node input : node.getInputs()) {
            dfs(input, visited, topoOrder);
        }
        
        topoOrder.add(node);
    }
    
    @Override
    public Tensor[] predict(StatesCache cache, Tensor... inputs) {
        return new Tensor[0];
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
