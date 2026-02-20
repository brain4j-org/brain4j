package org.brain4j.core.layer;

import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class Node {

    private final Layer layer;
    private final List<Node> inputs;
    private List<Shape> outputShapes;

    public Node(Layer layer, List<Node> inputs) {
        this.layer = layer;
        this.inputs = inputs;
    }

    public void build() {
        List<Shape> inputShapes = inputs.stream()
            .flatMap(n -> n.getOutputShapes().stream())
            .toList();

        this.outputShapes = layer.inferOutputShapes(inputShapes);
    }

    public Tensor[] forward(StatesCache cache, Map<Node, Tensor[]> computed) {
        List<Tensor> tensors = new ArrayList<>();

        for (Node input : inputs) {
            tensors.addAll(Arrays.asList(computed.get(input)));
        }

        Tensor[] out = layer.forward(cache, tensors.toArray(Tensor[]::new));
        computed.put(this, out);
        
        return out;
    }
    
    public Layer getLayer() {
        return layer;
    }
    
    public List<Node> getInputs() {
        return inputs;
    }
    
    public List<Shape> getOutputShapes() {
        return outputShapes;
    }
}