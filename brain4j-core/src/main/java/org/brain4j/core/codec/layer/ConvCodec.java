package org.brain4j.core.codec.layer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.JsonCodec;
import org.brain4j.core.layer.impl.ConvLayer;

public class ConvCodec implements JsonCodec<ConvLayer> {
    
    @Override
    public String type() {
        return "conv_2d";
    }
    
    @Override
    public Class<ConvLayer> targetClass() {
        return ConvLayer.class;
    }
    
    @Override
    public void write(ConvLayer conv, ObjectNode out) {
        out.put("filters", conv.config().filters());
        out.put("channels", conv.channels());
        out.put("kernel_width", conv.config().kernelWidth());
        out.put("kernel_height", conv.config().kernelHeight());
        out.put("padding", conv.config().padding());
        out.put("stride", conv.config().stride());
    }
    
    @Override
    public ConvLayer parse(JsonNode in) {
        int filters = in.get("filters").asInt();
        int kernelWidth = in.get("kernel_width").asInt();
        int kernelHeight = in.get("kernel_height").asInt();
        int stride = in.get("stride").asInt(1);
        
        return new ConvLayer(filters, kernelWidth, kernelHeight, stride);
    }
}
