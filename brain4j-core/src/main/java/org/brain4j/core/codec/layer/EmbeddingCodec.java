package org.brain4j.core.codec.layer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.JsonCodec;
import org.brain4j.core.layer.impl.transformer.EmbeddingLayer;

public class EmbeddingCodec implements JsonCodec<EmbeddingLayer> {

    @Override
    public String type() {
        return "embedding";
    }

    @Override
    public Class<EmbeddingLayer> targetClass() {
        return EmbeddingLayer.class;
    }

    @Override
    public void write(EmbeddingLayer layer, ObjectNode out) {
        out.put("vocab_size", layer.config().vocabSize());
        out.put("embedding_dim", layer.config().embeddingDim());
    }

    @Override
    public EmbeddingLayer parse(JsonNode in) {
        int vocab = in.get("vocab_size").asInt();
        int dim = in.get("embedding_dim").asInt();
        
        return new EmbeddingLayer(vocab, dim);
    }
}
