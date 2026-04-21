package org.brain4j.examples.dag;

import org.brain4j.core.layer.Node;
import org.brain4j.core.layer.impl.DenseLayer;
import org.brain4j.core.model.impl.Graph;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.impl.ReLU;
import org.brain4j.math.activation.impl.Sigmoid;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

public class GraphExample {
    
    public static void main(String[] args) {
        new GraphExample().start();
    }
    
    private void start() {
        Node input = Node.input(Shape.of(2));
        
        Node d1 = new DenseLayer(16, new ReLU()).apply(input);
        Node d2 = new DenseLayer(16, new ReLU()).apply(d1);
        Node out = new DenseLayer(1, new Sigmoid()).apply(d2);
        
        Graph model = Graph.of(out);
        Tensor prediction = model.predict(Tensors.ones(1, 2));
        System.out.println(prediction);
    }
}
