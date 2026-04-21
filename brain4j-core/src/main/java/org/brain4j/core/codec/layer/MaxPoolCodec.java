package org.brain4j.core.codec.layer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.Codec;
import org.brain4j.core.layer.impl.MaxPoolLayer;

public class MaxPoolCodec implements Codec<MaxPoolLayer> {

    @Override
    public String type() {
        return "max_pool";
    }

    @Override
    public Class<MaxPoolLayer> targetClass() {
        return MaxPoolLayer.class;
    }

    @Override
    public void write(MaxPoolLayer layer, ObjectNode out) {
        out.put("stride", layer.stride());
        out.put("window_height", layer.windowHeight());
        out.put("window_width", layer.windowWidth());
    }

    @Override
    public MaxPoolLayer parse(JsonNode in) {
        int stride = in.get("stride").asInt();
        int wh = in.get("window_height").asInt();
        int ww = in.get("window_width").asInt();
        
        return new MaxPoolLayer(stride, wh, ww);
    }
}
