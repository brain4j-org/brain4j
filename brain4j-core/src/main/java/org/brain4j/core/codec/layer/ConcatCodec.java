package org.brain4j.core.codec.layer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.JsonCodec;
import org.brain4j.core.layer.impl.ConcatLayer;
import org.brain4j.core.layer.impl.DropoutLayer;

public class ConcatCodec implements JsonCodec<ConcatLayer> {

    @Override
    public String type() {
        return "concat";
    }

    @Override
    public Class<ConcatLayer> targetClass() {
        return ConcatLayer.class;
    }

    @Override
    public void write(ConcatLayer concatLayer, ObjectNode out) {
        out.put("dimension", concatLayer.config().dimension());
    }

    @Override
    public ConcatLayer parse(JsonNode in) {
        int dim = in.get("dimension").asInt();
        return new ConcatLayer(dim);
    }
}
