package org.brain4j.core.codec.weightinit;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.Codec;
import org.brain4j.math.weightsinit.impl.NormalInit;

public class NormalInitCodec implements Codec<NormalInit> {
    
    @Override
    public String type() {
        return "normal";
    }
    
    @Override
    public Class<NormalInit> targetClass() {
        return NormalInit.class;
    }
    
    @Override
    public void write(NormalInit normalInit, ObjectNode out) {
    }
    
    @Override
    public NormalInit parse(JsonNode in) {
        return new NormalInit();
    }
}