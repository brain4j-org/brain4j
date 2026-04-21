package org.brain4j.core.codec.layer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.Codec;
import org.brain4j.core.layer.impl.transformer.TransformerDecoder;

public class TransformerDecoderCodec implements Codec<TransformerDecoder> {

    @Override
    public String type() {
        return "transformer_decoder";
    }

    @Override
    public Class<TransformerDecoder> targetClass() {
        return TransformerDecoder.class;
    }

    @Override
    public void write(TransformerDecoder layer, ObjectNode out) {
        out.put("num_heads", layer.numHeads());
        out.put("embedding_dim", layer.embeddingDim());
        out.put("dropout", layer.dropoutRate());
    }

    @Override
    public TransformerDecoder parse(JsonNode in) {
        int heads = in.get("num_heads").asInt();
        int dim = in.get("embedding_dim").asInt();
        double dropout = in.get("dropout").asDouble();

        return new TransformerDecoder(heads, dim, dropout);
    }
}
