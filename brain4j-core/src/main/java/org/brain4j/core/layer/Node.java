package org.brain4j.core.layer;

import org.brain4j.core.layer.newimpl.InputLayer;
import org.brain4j.math.Copyable;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

import java.util.*;
import java.util.random.RandomGenerator;

public class Node implements Copyable<Node> {

    private final Layer layer;
    private final List<Node> inputs;
    private List<Shape> outputShapes;

    public Node(Layer layer, List<Node> inputs) {
        this.layer = layer;
        this.inputs = inputs;
    }
    
    public static Node input(Shape shape) {
        return new Node(new InputLayer(shape), List.of());
    }
    
    @Override
    public Node copy() {
        return new Node(layer.copy(), inputs);
    }
    
    public void build() {
        List<Shape> inputShapes = inferInputShapes();
        this.outputShapes = layer.inferOutputShapes(inputShapes);

        if (layer.frozen()) return;

        layer.build(inputShapes);
    }
    
    public void initWeights(int seed) {
        List<Shape> inputShapes = inferInputShapes();
        RandomGenerator rng = new SplittableRandom(seed);
        
        layer.initWeights(inputShapes, rng);
        layer.initAutoGrad();
    }
    
    public List<Shape> inferInputShapes() {
        return inputs.stream()
            .flatMap(n -> n.getOutputShapes().stream())
            .toList();
    }
    
    public Tensor[] forward(StatesCache cache, Map<Node, Tensor[]> computed) {
        List<Tensor> tensors = new ArrayList<>();
        
        for (Node input : inputs) {
            // outputs of the previous node
            Tensor[] inTensors = computed.get(input);
            tensors.addAll(Arrays.asList(inTensors));
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