package org.brain4j.core.model;

import org.brain4j.core.layer.Layer;

import java.util.List;

public interface ModelBlock {
    void appendTo(List<Layer> layers);
}
