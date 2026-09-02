package org.brain4j.core.codec.scaler;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.JsonCodec;
import org.brain4j.math.scaler.impl.ZScoreScaler;

public class ZScoreScalerCodec implements JsonCodec<ZScoreScaler> {

    @Override
    public String type() {
        return "z_score";
    }

    @Override
    public Class<ZScoreScaler> targetClass() {
        return ZScoreScaler.class;
    }

    @Override
    public void write(ZScoreScaler scaler, ObjectNode out) {
        out.put("mean", scaler.mean());
        out.put("std", scaler.std());
    }

    @Override
    public ZScoreScaler parse(JsonNode in) {
        float mean = (float) in.get("mean").asDouble();
        float std = (float) in.get("std").asDouble();
        return new ZScoreScaler(mean, std);
    }
}
