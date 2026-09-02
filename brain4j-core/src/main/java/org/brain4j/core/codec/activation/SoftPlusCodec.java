package org.brain4j.core.codec.activation;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.JsonCodec;
import org.brain4j.math.activation.impl.SoftPlus;

public class SoftPlusCodec implements JsonCodec<SoftPlus> {
    
    @Override
    public String type() {
        return "softplus";
    }
    
    @Override
    public Class<SoftPlus> targetClass() {
        return SoftPlus.class;
    }
    
    @Override
    public void write(SoftPlus softplus, ObjectNode out) {
    }
    
    @Override
    public SoftPlus parse(JsonNode in) {
        return new SoftPlus();
    }
}