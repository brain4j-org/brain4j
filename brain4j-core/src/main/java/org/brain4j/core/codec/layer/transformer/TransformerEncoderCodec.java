package org.brain4j.core.codec.layer.transformer;

import com.fasterxml.jackson.databind.JsonNode;
import org.brain4j.core.layer.impl.transformer.Transformer;

public class TransformerEncoderCodec implements TransformerCodec<Transformer.Encoder> {

    @Override
    public String type() {
        return "transformer_encoder";
    }

    @Override
    public Class<Transformer.Encoder> targetClass() {
        return Transformer.Encoder.class;
    }

    @Override
    public Transformer.Encoder parse(JsonNode in) {
        return new Transformer.Encoder(readConfig(in));
    }
}
