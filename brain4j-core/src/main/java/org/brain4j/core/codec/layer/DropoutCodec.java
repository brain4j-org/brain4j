package org.brain4j.core.codec.layer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.JsonCodec;
import org.brain4j.core.layer.impl.DropoutLayer;

public class DropoutCodec implements JsonCodec<DropoutLayer> {

    @Override
    public String type() {
        return "dropout";
    }

    @Override
    public Class<DropoutLayer> targetClass() {
        return DropoutLayer.class;
    }

    @Override
    public void write(DropoutLayer dropoutLayer, ObjectNode out) {
        out.put("rate", dropoutLayer.config().dropoutRate());
    }

    @Override
    public DropoutLayer parse(JsonNode in) {
        double rate = in.get("rate").asDouble();
        return new DropoutLayer(rate);
    }
}
