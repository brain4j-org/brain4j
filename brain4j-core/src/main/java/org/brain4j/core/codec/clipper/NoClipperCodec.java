package org.brain4j.core.codec.clipper;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.Codec;
import org.brain4j.math.clipper.impl.NoClipper;

public class NoClipperCodec implements Codec<NoClipper> {
    
    @Override
    public String type() {
        return "none";
    }
    
    @Override
    public Class<NoClipper> targetClass() {
        return NoClipper.class;
    }
    
    @Override
    public void write(NoClipper noClipper, ObjectNode out) {
    }
    
    @Override
    public NoClipper parse(JsonNode in) {
        return new NoClipper();
    }
}
