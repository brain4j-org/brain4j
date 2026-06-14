package org.brain4j.core.codec.layer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.Codec;
import org.brain4j.core.layer.impl.NormLayer;

public class NormCodec implements Codec<NormLayer> {
    
    @Override
    public String type() {
        return "norm";
    }
    
    @Override
    public Class<NormLayer> targetClass() {
        return NormLayer.class;
    }
    
    @Override
    public void write(NormLayer normLayer, ObjectNode out) {
        out.put("epsilon", normLayer.config().epsilon());
    }
    
    @Override
    public NormLayer parse(JsonNode in) {
        double epsilon = in.get("epsilon").asDouble();
        return new NormLayer(epsilon);
    }
}
