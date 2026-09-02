package org.brain4j.examples.dag;

import org.brain4j.core.graph.impl.MermaidExporter;
import org.brain4j.core.layer.Node;
import org.brain4j.core.layer.impl.ConcatLayer;
import org.brain4j.core.layer.impl.DenseLayer;
import org.brain4j.core.layer.impl.utility.SelectLayer;
import org.brain4j.core.model.impl.Graph;
import org.brain4j.math.activation.impl.ReLU;
import org.brain4j.math.activation.impl.Sigmoid;
import org.brain4j.math.tensor.Shape;

public class GraphExample {
    
    public static void main(String[] args) {
        new GraphExample().start();
    }
    
    private void start() {
        Node input1 = Node.input(Shape.of(2));
        Node input2 = Node.input(Shape.of(5));
        
        Node d11 = new DenseLayer(16, new ReLU()).apply(input1);
        Node d12 = new DenseLayer(12, new ReLU()).apply(d11);
        
        Node d21 = new DenseLayer(12, new ReLU()).apply(input2);
        
        Node result = new ConcatLayer().apply(d12, d21);
        Node out = new DenseLayer(1, new Sigmoid()).apply(result);
        
        Graph model = Graph.of(out);
        MermaidExporter exporter = new MermaidExporter();
        System.out.println(exporter.export(model));
    }
}
