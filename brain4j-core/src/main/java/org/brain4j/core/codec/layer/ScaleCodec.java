package org.brain4j.core.codec.layer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.JsonCodec;
import org.brain4j.core.layer.impl.ScaleLayer;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.scaler.FeatureScaler;

import java.util.HashSet;
import java.util.Set;

import static org.brain4j.core.importing.io.LayerIO.SCALER_CODECS;
import static org.brain4j.core.importing.format.impl.BrainFormat.MAPPER;

public class ScaleCodec implements JsonCodec<ScaleLayer> {

    @Override
    public String type() {
        return "scale";
    }

    @Override
    public Class<ScaleLayer> targetClass() {
        return ScaleLayer.class;
    }

    @Override
    public void write(ScaleLayer scaleLayer, ObjectNode out) {
        ObjectNode config = MAPPER.createObjectNode();

        FeatureScaler scaler = scaleLayer.config().scaler();
        JsonCodec<FeatureScaler> codec = (JsonCodec<FeatureScaler>) SCALER_CODECS.get(scaler.getClass());

        Set<Integer> ints = scaleLayer.config().enabledInputs();

        codec.write(scaler, config);

        out.put("scaler", codec.type());

        if (ints != null) {
            ArrayNode enabled = MAPPER.createArrayNode();
            for (int i : ints) enabled.add(i);
            out.set("enabled", enabled);
        }

        out.set("config", config);
    }

    @Override
    public ScaleLayer parse(JsonNode in) {
        String scalerType = in.get("scaler").asText();
        JsonNode config = in.get("config");

        JsonCodec<FeatureScaler> codec = (JsonCodec<FeatureScaler>) SCALER_CODECS.get(scalerType);
        FeatureScaler scaler = codec.parse(config);

        JsonNode enabled = in.get("enabled");

        if (enabled == null || enabled.isNull()) {
            return new ScaleLayer(scaler);
        }

        if (!enabled.isArray()) {
            throw Commons.illegalArgument("Enabled must be an array");
        }

        Set<Integer> enabledSet = new HashSet<>();

        for (int i = 0; i < enabled.size(); i++) {
            enabledSet.add(enabled.get(i).asInt());
        }

        return new ScaleLayer(scaler, enabledSet);
    }
}
