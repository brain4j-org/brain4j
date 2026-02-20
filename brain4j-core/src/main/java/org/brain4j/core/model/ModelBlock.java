package org.brain4j.core.model;

import org.brain4j.core.layer.Layer0;

import java.util.List;

public interface ModelBlock {
    void appendTo(List<Layer0> layers);
}
