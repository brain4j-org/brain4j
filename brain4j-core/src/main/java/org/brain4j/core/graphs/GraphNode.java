package org.brain4j.core.graphs;

import org.brain4j.core.layer.Layer;

import java.util.List;

public record GraphNode(String name, Layer layer, List<GraphNode> inputs) {
}