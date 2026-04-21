package org.brain4j.core.codec.layer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.Codec;
import org.brain4j.core.layer.impl.transformer.TransformerEncoder;
import org.brain4j.core.layer.impl.transformer.NormType;
import org.brain4j.math.activation.Activations;

public class TransformerEncoderCodec implements Codec<TransformerEncoder> {

    @Override
    public String type() {
        return "transformer_encoder";
    }

    @Override
    public Class<TransformerEncoder> targetClass() {
        return TransformerEncoder.class;
    }

    @Override
    public void write(TransformerEncoder layer, ObjectNode out) {
        out.put("num_heads", layer.numHeads());
        out.put("embedding_dim", layer.embeddingDim());
        out.put("dropout", layer.dropoutRate());
    }

    @Override
    public TransformerEncoder parse(JsonNode in) {
        int heads = in.get("num_heads").asInt();
        int dim = in.get("embedding_dim").asInt();
        double dropout = in.get("dropout").asDouble();
        
        // TODO: finish this?
        return new TransformerEncoder(heads, dim, dropout);
    }
}
