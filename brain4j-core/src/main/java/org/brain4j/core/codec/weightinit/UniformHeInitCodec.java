package org.brain4j.core.codec.weightinit;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.JsonCodec;
import org.brain4j.math.weightsinit.impl.UniformHeInit;

public class UniformHeInitCodec implements JsonCodec<UniformHeInit> {
    
    @Override
    public String type() {
        return "uniform_he";
    }
    
    @Override
    public Class<UniformHeInit> targetClass() {
        return UniformHeInit.class;
    }
    
    @Override
    public void write(UniformHeInit uniformHeInit, ObjectNode out) {
    }
    
    @Override
    public UniformHeInit parse(JsonNode in) {
        return new UniformHeInit();
    }
}
