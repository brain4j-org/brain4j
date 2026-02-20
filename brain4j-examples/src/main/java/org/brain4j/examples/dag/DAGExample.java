package org.brain4j.examples.dag;

import org.brain4j.core.layer.Node;
import org.brain4j.core.layer.newimpl.DenseLayer;
import org.brain4j.core.model.impl.DAG;
import org.brain4j.math.activation.impl.ReLU;
import org.brain4j.math.activation.impl.Softmax;
import org.brain4j.math.tensor.Shape;

import java.util.List;

public class DAGExample {
    
    public static void main(String[] args) {
        new DAGExample().start();
    }
    
    private void start() {
        Node input = Node.input(Shape.of(28 * 28));
        
        Node d1 = new DenseLayer(128, new ReLU()).apply(input);
        Node d2 = new DenseLayer(64, new ReLU()).apply(d1);
        Node out = new DenseLayer(10, new Softmax()).apply(d2);
        
        DAG model = DAG.of(out);
        System.out.println("GG!");
        
        System.out.println(d1.getLayer().getParam("weights"));
    }
}
