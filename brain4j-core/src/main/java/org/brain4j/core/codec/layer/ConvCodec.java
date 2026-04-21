package org.brain4j.core.codec.layer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.Codec;
import org.brain4j.core.layer.impl.ConvLayer;

public class ConvCodec implements Codec<ConvLayer> {
    
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
        out.put("filters", conv.filters());
        out.put("channels", conv.channels());
        out.put("kernel_width", conv.kernelWidth());
        out.put("kernel_height", conv.kernelHeight());
        out.put("padding", conv.padding());
        out.put("stride", conv.stride());
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
