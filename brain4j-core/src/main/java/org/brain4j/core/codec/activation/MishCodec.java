package org.brain4j.core.codec.activation;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.Codec;
import org.brain4j.math.activation.impl.Mish;

public class MishCodec implements Codec<Mish> {
    
    @Override
    public String type() {
        return "mish";
    }
    
    @Override
    public Class<Mish> targetClass() {
        return Mish.class;
    }
    
    @Override
    public void write(Mish mish, ObjectNode out) {
    }
    
    @Override
    public Mish parse(JsonNode in) {
        return new Mish();
    }
}