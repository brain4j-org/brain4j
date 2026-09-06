package org.brain4j.core.codec.clipper;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.JsonCodec;
import org.brain4j.math.clipper.impl.L2Clipper;

public class L2ClipperCodec implements JsonCodec<L2Clipper> {
    
    @Override
    public String type() {
        return "l2";
    }
    
    @Override
    public Class<L2Clipper> targetClass() {
        return L2Clipper.class;
    }
    
    @Override
    public void write(L2Clipper l2Clipper, ObjectNode out) {
        out.put("scale", l2Clipper.scale());
    }
    
    @Override
    public L2Clipper parse(JsonNode in) {
        double scale = in.get("scale").asDouble(1.0);
        return new L2Clipper(scale);
    }
}
