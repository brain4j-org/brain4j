package org.brain4j.core.codec.activation;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.Codec;
import org.brain4j.math.activation.impl.LeakyReLU;

public class LeakyReLUCodec implements Codec<LeakyReLU> {
    
    @Override
    public String type() {
        return "leaky_relu";
    }
    
    @Override
    public Class<LeakyReLU> targetClass() {
        return LeakyReLU.class;
    }
    
    @Override
    public void write(LeakyReLU leakyReLU, ObjectNode out) {
        out.put("alpha", leakyReLU.alpha());
    }
    
    @Override
    public LeakyReLU parse(JsonNode in) {
        double alpha = in.get("alpha").asDouble(0.01);
        return new LeakyReLU(alpha);
    }
}
