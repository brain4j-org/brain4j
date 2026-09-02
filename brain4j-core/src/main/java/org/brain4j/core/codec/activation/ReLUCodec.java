package org.brain4j.core.codec.activation;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.JsonCodec;
import org.brain4j.math.activation.impl.ReLU;

public class ReLUCodec implements JsonCodec<ReLU> {
    
    @Override
    public String type() {
        return "relu";
    }
    
    @Override
    public Class<ReLU> targetClass() {
        return ReLU.class;
    }
    
    @Override
    public void write(ReLU relu, ObjectNode out) {
    }
    
    @Override
    public ReLU parse(JsonNode in) {
        return new ReLU();
    }
}