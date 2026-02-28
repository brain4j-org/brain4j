package org.brain4j.core.codec.activation;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.Codec;
import org.brain4j.math.activation.impl.GELU;

public class GELUCodec implements Codec<GELU> {
    
    @Override
    public String type() {
        return "gelu";
    }
    
    @Override
    public Class<GELU> targetClass() {
        return GELU.class;
    }
    
    @Override
    public void write(GELU gelu, ObjectNode out) {
    }
    
    @Override
    public GELU parse(JsonNode in) {
        return new GELU();
    }
}
