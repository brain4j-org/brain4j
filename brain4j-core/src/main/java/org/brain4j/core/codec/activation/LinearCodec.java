package org.brain4j.core.codec.activation;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.JsonCodec;
import org.brain4j.math.activation.impl.Linear;

public class LinearCodec implements JsonCodec<Linear> {
    
    @Override
    public String type() {
        return "linear";
    }
    
    @Override
    public Class<Linear> targetClass() {
        return Linear.class;
    }
    
    @Override
    public void write(Linear linear, ObjectNode out) {
    }
    
    @Override
    public Linear parse(JsonNode in) {
        return new Linear();
    }
}
