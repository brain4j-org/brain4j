package org.brain4j.core.codec.activation;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.Codec;
import org.brain4j.math.activation.impl.Softmax;

public class SoftmaxCodec implements Codec<Softmax> {
    
    @Override
    public String type() {
        return "softmax";
    }
    
    @Override
    public Class<Softmax> targetClass() {
        return Softmax.class;
    }
    
    @Override
    public void write(Softmax softmax, ObjectNode out) {
        out.put("temperature", softmax.temperature());
    }
    
    @Override
    public Softmax parse(JsonNode in) {
        double temperature = in.get("temperature").asDouble();
        return new Softmax(temperature);
    }
}
