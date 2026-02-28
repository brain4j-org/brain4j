package org.brain4j.core.codec.clipper;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.Codec;
import org.brain4j.math.clipper.impl.HardClipper;

public class HardClipperCodec implements Codec<HardClipper> {
    
    @Override
    public String type() {
        return "clamp";
    }
    
    @Override
    public Class<HardClipper> targetClass() {
        return HardClipper.class;
    }
    
    @Override
    public void write(HardClipper hardClipper, ObjectNode out) {
        out.put("bound", hardClipper.bound());
    }
    
    @Override
    public HardClipper parse(JsonNode in) {
        double bound = in.get("bound").asDouble(5.0);
        return new HardClipper(bound);
    }
}
