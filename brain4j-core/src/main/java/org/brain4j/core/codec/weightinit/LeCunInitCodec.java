package org.brain4j.core.codec.weightinit;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.JsonCodec;
import org.brain4j.math.weightsinit.impl.LeCunInit;

public class LeCunInitCodec implements JsonCodec<LeCunInit> {
    
    @Override
    public String type() {
        return "lecun";
    }
    
    @Override
    public Class<LeCunInit> targetClass() {
        return LeCunInit.class;
    }
    
    @Override
    public void write(LeCunInit leCunInit, ObjectNode out) {
    }
    
    @Override
    public LeCunInit parse(JsonNode in) {
        return new LeCunInit();
    }
}
