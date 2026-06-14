package org.brain4j.core.codec.layer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.Codec;
import org.brain4j.core.layer.impl.DenseLayer;

public class DenseCodec implements Codec<DenseLayer> {
    
    @Override
    public String type() {
        return "dense";
    }
    
    @Override
    public Class<DenseLayer> targetClass() {
        return DenseLayer.class;
    }
    
    @Override
    public void write(DenseLayer denseLayer, ObjectNode out) {
        out.put("dimension", denseLayer.config().outDimension());
    }
    
    @Override
    public DenseLayer parse(JsonNode in) {
        int dim = in.get("dimension").asInt();
        return new DenseLayer(dim);
    }
}
