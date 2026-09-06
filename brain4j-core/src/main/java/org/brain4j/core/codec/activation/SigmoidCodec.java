package org.brain4j.core.codec.activation;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.JsonCodec;
import org.brain4j.math.activation.impl.Sigmoid;

public class SigmoidCodec implements JsonCodec<Sigmoid> {
    
    @Override
    public String type() {
        return "sigmoid";
    }
    
    @Override
    public Class<Sigmoid> targetClass() {
        return Sigmoid.class;
    }
    
    @Override
    public void write(Sigmoid sigmoid, ObjectNode out) {
    }
    
    @Override
    public Sigmoid parse(JsonNode in) {
        return new Sigmoid();
    }
}