package org.brain4j.math.commons;

import com.fasterxml.jackson.databind.node.ObjectNode;

public interface JsonAdapter {
    void serialize(ObjectNode object);
    void deserialize(ObjectNode object);
}
