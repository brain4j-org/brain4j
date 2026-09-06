package org.brain4j.core.codec.layer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.JsonCodec;
import org.brain4j.core.layer.impl.LiquidLayer;

public class LiquidCodec implements JsonCodec<LiquidLayer> {

    @Override
    public String type() {
        return "liquid";
    }

    @Override
    public Class<LiquidLayer> targetClass() {
        return LiquidLayer.class;
    }

    @Override
    public void write(LiquidLayer layer, ObjectNode out) {
        LiquidLayer.Config config = layer.config();
        out.put("dimension", config.hiddenDimension());
        out.put("solver_steps", config.solverSteps());
        out.put("tau_min", config.tauMin());
        out.put("return_sequences", config.returnSequences());
    }

    @Override
    public LiquidLayer parse(JsonNode in) {
        return new LiquidLayer(
            in.get("dimension").asInt(),
            in.get("solver_steps").asInt(),
            in.get("tau_min").asDouble(),
            in.get("return_sequences").asBoolean()
        );
    }
}
