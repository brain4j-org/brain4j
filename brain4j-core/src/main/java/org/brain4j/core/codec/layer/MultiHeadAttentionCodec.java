package org.brain4j.core.codec.layer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.Codec;
import org.brain4j.core.layer.impl.transformer.MultiHeadAttention;

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
        out.put("head_count", layer.headCount());
        out.put("embedding_dim", layer.embeddingDim());
        out.put("attn_qkv_has_bias", layer.attnQkvHasBias());
        out.put("attn_out_has_bias", layer.attnOutHasBias());
    }

    @Override
    public MultiHeadAttention parse(JsonNode in) {
        int heads = in.get("head_count").asInt();
        int dim = in.get("embedding_dim").asInt();
        
        JsonNode qkv = in.get("attn_qkv_has_bias");
        JsonNode out = in.get("attn_out_has_bias");

        MultiHeadAttention mha = new MultiHeadAttention(heads, dim);

        if (qkv != null) mha.attnQkvHasBias(qkv.asBoolean());
        if (out != null) mha.attnOutHasBias(out.asBoolean());

        return mha;
    }
}
