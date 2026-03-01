package org.brain4j.core.codec.layer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.Codec;
import org.brain4j.core.layer.newimpl.LSTMLayer;

public class LSTMCodec implements Codec<LSTMLayer> {
    
    @Override
    public String type() {
        return "lstm";
    }
    
    @Override
    public Class<LSTMLayer> targetClass() {
        return LSTMLayer.class;
    }
    
    @Override
    public void write(LSTMLayer lstmLayer, ObjectNode out) {
        out.put("hidden_dimension", lstmLayer.hiddenDimension());
        out.put("return_sequences", lstmLayer.returnSequences());
    }
    
    @Override
    public LSTMLayer parse(JsonNode in) {
        int hiddenDimension = in.get("hidden_dimension").asInt();
        boolean returnSequences = in.get("return_sequences").asBoolean();
        return new LSTMLayer(hiddenDimension, returnSequences);
    }
}
