package org.brain4j.core.codec.layer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.Codec;
import org.brain4j.core.layer.impl.transformer.attention.MultiHeadAttention;

public class MultiHeadAttentionCodec implements Codec<MultiHeadAttention> {

    @Override
    public String type() {
        return "multi_head_attention";
    }

    @Override
    public Class<MultiHeadAttention> targetClass() {
        return MultiHeadAttention.class;
    }

    @Override
    public void write(MultiHeadAttention layer, ObjectNode out) {
        out.put("head_count", layer.config().heads());
        out.put("embedding_dim", layer.config().embedDim());
        out.put("qkv_bias", layer.config().qkvBias());
        out.put("out_bias", layer.config().outBias());
    }

    @Override
    public MultiHeadAttention parse(JsonNode in) {
        int heads = in.get("head_count").asInt();
        int dim = in.get("embedding_dim").asInt();

        boolean qkv = in.get("qkv_bias").asBoolean();
        boolean out = in.get("out_bias").asBoolean();

        var config = new MultiHeadAttention.Config(dim, heads, qkv, out);
        return new MultiHeadAttention(config);
    }
}
