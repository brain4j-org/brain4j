package org.brain4j.core.codec.layer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.JsonCodec;
import org.brain4j.core.layer.impl.utility.SqueezeLayer;

public class SqueezeCodec implements JsonCodec<SqueezeLayer> {

    @Override
    public String type() {
        return "squeeze";
    }

    @Override
    public Class<SqueezeLayer> targetClass() {
        return SqueezeLayer.class;
    }

    @Override
    public void write(SqueezeLayer layer, ObjectNode out) {
        out.put("dimension", layer.config().dimension());
    }

    @Override
    public SqueezeLayer parse(JsonNode in) {
        int dim = in.get("dimension").asInt();
        return new SqueezeLayer(dim);
    }
}
