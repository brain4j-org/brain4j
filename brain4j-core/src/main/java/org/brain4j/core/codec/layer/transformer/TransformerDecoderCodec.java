package org.brain4j.core.codec.layer.transformer;

import com.fasterxml.jackson.databind.JsonNode;
import org.brain4j.core.layer.impl.transformer.Transformer;

public class TransformerDecoderCodec implements TransformerCodec<Transformer.Decoder> {

    @Override
    public String type() {
        return "transformer_decoder";
    }

    @Override
    public Class<Transformer.Decoder> targetClass() {
        return Transformer.Decoder.class;
    }

    @Override
    public Transformer.Decoder parse(JsonNode in) {
        return new Transformer.Decoder(readConfig(in));
    }
}
