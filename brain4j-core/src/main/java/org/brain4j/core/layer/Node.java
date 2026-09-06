package org.brain4j.core.layer;

import org.brain4j.core.importing.io.LayerIO;
import org.brain4j.core.layer.impl.InputLayer;
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
        return copy(new HashMap<>());
    }

    public Node copy(Map<Node, Node> cache) {
        if (cache.containsKey(this)) {
            return cache.get(this);
        }
        Node copy = new Node(layer.copy(), new ArrayList<>());
        cache.put(this, copy);
        for (Node in : inputs) {
            copy.inputs.add(in.copy(cache));
        }
        copy.outputShapes = this.outputShapes == null ? null : new ArrayList<>(this.outputShapes);
        return copy;
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

        if (layer.frozen()) return;
        
        layer.initWeights(inputShapes, rng);
        layer.initAutoGrad();
    }
    
    public List<Shape> inferInputShapes() {
        return inputs.stream()
            .flatMap(n -> n.outputShapes().stream())
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
    
    public Layer layer() {
        return layer;
    }
    
    public List<Node> inputs() {
        return inputs;
    }
    
    public List<Shape> outputShapes() {
        return outputShapes;
    }
    
    public String name() {
        return LayerIO.LAYER_CODECS.get(layer.getClass()).type();
    }
}