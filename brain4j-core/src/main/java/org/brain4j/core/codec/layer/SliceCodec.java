package org.brain4j.core.codec.layer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.Codec;
import org.brain4j.core.layer.impl.utility.SliceLayer;
import org.brain4j.math.commons.Range;

public class SliceCodec implements Codec<SliceLayer> {

    @Override
    public String type() {
        return "slice";
    }

    @Override
    public Class<SliceLayer> targetClass() {
        return SliceLayer.class;
    }

    @Override
    public void write(SliceLayer layer, ObjectNode out) {
        // TODO
    }

    @Override
    public SliceLayer parse(JsonNode in) {
        // TODO
        return null;
    }
}
