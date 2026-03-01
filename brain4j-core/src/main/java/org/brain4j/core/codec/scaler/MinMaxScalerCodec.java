package org.brain4j.core.codec.scaler;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.Codec;
import org.brain4j.math.scaler.impl.MinMaxScaler;

public class MinMaxScalerCodec implements Codec<MinMaxScaler> {

    @Override
    public String type() {
        return "min_max";
    }

    @Override
    public Class<MinMaxScaler> targetClass() {
        return MinMaxScaler.class;
    }

    @Override
    public void write(MinMaxScaler scaler, ObjectNode out) {
        out.put("range_min", scaler.rangeMin());
        out.put("range_max", scaler.rangeMax());
        out.put("data_min", scaler.dataMin());
        out.put("data_max", scaler.dataMax());
    }

    @Override
    public MinMaxScaler parse(JsonNode in) {
        float rangeMin = (float) in.get("range_min").asDouble();
        float rangeMax = (float) in.get("range_max").asDouble();
        float dataMin = (float) in.get("data_min").asDouble();
        float dataMax = (float) in.get("data_max").asDouble();
        return new MinMaxScaler(rangeMin, rangeMax, dataMin, dataMax);
    }
}
