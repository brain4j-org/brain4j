package org.brain4j.core.codec.activation;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.Codec;
import org.brain4j.math.activation.impl.Tanh;

public class TanhCodec implements Codec<Tanh> {
    
    @Override
    public String type() {
        return "tanh";
    }
    
    @Override
    public Class<Tanh> targetClass() {
        return Tanh.class;
    }
    
    @Override
    public void write(Tanh tanh, ObjectNode out) {
    }
    
    @Override
    public Tanh parse(JsonNode in) {
        return new Tanh();
    }
}