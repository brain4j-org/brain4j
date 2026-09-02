package org.brain4j.core.codec.activation;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.JsonCodec;
import org.brain4j.math.activation.impl.Swish;

public class SwishCodec implements JsonCodec<Swish> {
    
    @Override
    public String type() {
        return "swish";
    }
    
    @Override
    public Class<Swish> targetClass() {
        return Swish.class;
    }
    
    @Override
    public void write(Swish swish, ObjectNode out) {
    }
    
    @Override
    public Swish parse(JsonNode in) {
        return new Swish();
    }
}