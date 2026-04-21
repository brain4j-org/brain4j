package org.brain4j.core.graph.impl;

import org.brain4j.core.graph.GraphExporter;
import org.brain4j.core.layer.Node;
import org.brain4j.core.model.impl.Graph;
import org.brain4j.core.utils.CodeWriter;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.tensor.Shape;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class MermaidExporter implements GraphExporter {
    
    private final Direction direction;
    
    public MermaidExporter() {
        this.direction = Direction.TOP_DOWN;
    }
    
    public MermaidExporter(Direction direction) {
        this.direction = direction;
    }
    
    @Override
    public String export(Graph model) {
        CodeWriter writer = new CodeWriter();
        
        writer.writeLine("flowchart " + direction.id());
        writer.indent();
        
        Map<Node, String> ids = new HashMap<>();
        int counter = 0;
        
        for (Node in : model.input()) {
            ids.put(in, "n" + counter++);
        }
        
        for (Node node : model.topology()) {
            ids.put(node, "n" + counter++);
        }
        
        List<Node> inputNodes = model.input();
        
        for (int i = 0; i < inputNodes.size(); i++) {
            String uid = ids.get(inputNodes.get(i));
            writer.writeLine("%s[%s]".formatted(uid, "Input " + (i + 1)));
        }
        
        for (Node node : model.topology()) {
            String uid = ids.get(node);
            Activation activation = node.layer().activation();
            
            String name = Commons.capitalize(node.name()) + "\\n"
                + (activation != null ? activation.name() : "");
            
            writer.writeLine("%s[%s]".formatted(uid, name));
        }
        
        for (Node node : model.topology()) {
            String nodeUid = ids.get(node);
            
            for (Node input : node.inputs()) {
                String inputUid = ids.get(input);
                
                for (Shape inShape : input.outputShapes()) {
                    String label = format(inShape);
                    writer.writeLine("%s -->|\"%s\"| %s".formatted(inputUid, label, nodeUid));
                }
            }
        }
        
        return writer.toString();
    }
    
    private String format(Shape shape) {
        String tmp = Arrays.stream(shape.dims())
            .mapToObj(String::valueOf)
            .collect(Collectors.joining("x"));
        return "(Bx%s)".formatted(tmp);
    }
    
    public enum Direction {
        TOP_DOWN("TD"),
        DOWN_TOP("DT"),
        LEFT_RIGHT("LR"),
        RIGHT_LEFT("RL");
        
        private final String id;
        
        Direction(String id) {
            this.id = id;
        }
        
        public String id() {
            return id;
        }
    }
}
