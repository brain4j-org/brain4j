package org.brain4j.core.codec.layer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.JsonCodec;
import org.brain4j.core.layer.impl.transformer.PosEncodeLayer;

public class PosEncodeCodec implements JsonCodec<PosEncodeLayer> {

    @Override
    public String type() {
        return "pos_encode";
    }

    @Override
    public Class<PosEncodeLayer> targetClass() {
        return PosEncodeLayer.class;
    }

    @Override
    public void write(PosEncodeLayer layer, ObjectNode out) {
        out.put("length", layer.config().length());
        out.put("dimension", layer.config().dimension());
    }

    @Override
    public PosEncodeLayer parse(JsonNode in) {
        int length = in.get("length").asInt();
        int dim = in.get("dimension").asInt();
        return new PosEncodeLayer(length, dim);
    }
}
