package org.brain4j.core.codec.layer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.Codec;
import org.brain4j.core.layer.impl.utility.SliceLayer;
import org.brain4j.math.commons.Commons;
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
        ArrayNode shape = out.putArray("ranges");
        for (Range range : layer.ranges()) {
            ArrayNode node = shape.arrayNode();
            node.add(range.start());
            node.add(range.end());
            node.add(range.step());
            shape.add(node);
        }
    }

    @Override
    public SliceLayer parse(JsonNode in) {
        JsonNode jsonRanges = in.get("ranges");

        if (jsonRanges == null || !jsonRanges.isArray()) {
            throw Commons.illegalArgument("Ranges must be an array");
        }

        Range[] ranges = new Range[jsonRanges.size()];

        for (int i = 0; i < jsonRanges.size(); i++) {
            JsonNode node = jsonRanges.get(i);
            int start = node.get(0).intValue();
            int end = node.get(1).intValue();
            int step = node.get(3).intValue();
            ranges[i] = new Range(start, end, step);
        }

        return new SliceLayer(ranges);
    }
}
