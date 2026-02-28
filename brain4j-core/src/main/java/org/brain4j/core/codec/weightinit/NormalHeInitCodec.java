package org.brain4j.core.codec.weightinit;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.Codec;
import org.brain4j.math.weightsinit.impl.NormalHeInit;

public class NormalHeInitCodec implements Codec<NormalHeInit> {
    
    @Override
    public String type() {
        return "normal_he";
    }
    
    @Override
    public Class<NormalHeInit> targetClass() {
        return NormalHeInit.class;
    }
    
    @Override
    public void write(NormalHeInit normalHeInit, ObjectNode out) {
    }
    
    @Override
    public NormalHeInit parse(JsonNode in) {
        return new NormalHeInit();
    }
}
