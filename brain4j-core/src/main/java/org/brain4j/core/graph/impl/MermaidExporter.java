package org.brain4j.core.graph.impl;

import org.brain4j.core.graph.GraphExporter;
import org.brain4j.core.layer.Node;
import org.brain4j.core.model.impl.Graph;
import org.brain4j.core.utils.CodeWriter;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.tensor.Shape;

import java.util.Arrays;
import java.util.List;

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
        
        
        List<Node> inputNodes = model.input();
        List<Node> outputNodes = model.output();
        
        for (int i = 0; i < inputNodes.size(); i++) {
            String uid = String.valueOf(inputNodes.get(i).hashCode());
            
            writer.writeLine("%s[%s]".formatted(uid, "Input " + i));
        }
        
        for (Node node : model.topology()) {
            String uid = String.valueOf(node.hashCode());
            String name = Commons.capitalize(node.name()) + "\\n" + node.layer().activation().name();
            
            writer.writeLine("%s[%s]".formatted(uid, name));
        }
        
        for (Node node : model.topology()) {
            String nodeUid = String.valueOf(node.hashCode());
            
            for (Node input : node.inputs()) {
                String inputUid = String.valueOf(input.hashCode());
                
                List<Shape> inputOutShapes = input.outputShapes();
                List<Shape> nodeOutShapes = node.outputShapes();
                
                for (Shape inShape : inputOutShapes) {
                    String inShapeDims = Arrays.toString(inShape.dims())
                        .replace("[", "")
                        .replace("]", "");
                    
                    for (Shape outShape : nodeOutShapes) {
                        String outShapeDims = Arrays.toString(outShape.dims())
                            .replace("[", "")
                            .replace("]", "");
                        
                        String label = "In: (%s) > Out: (%s)".formatted(inShapeDims, outShapeDims);
                        
                        writer.writeLine("%s -->|\"%s\"| %s".formatted(inputUid, label, nodeUid));
                        
                    }
                }
            }
        }
        
        return writer.toString();
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
