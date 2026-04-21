package org.brain4j.core.codec.layer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.Codec;
import org.brain4j.core.layer.impl.transformer.MaskedMultiHeadAttention;
import org.brain4j.math.clipper.impl.HardClipper;

public class MaskedMultiHeadAttentionCodec implements Codec<MaskedMultiHeadAttention> {

    @Override
    public String type() {
        return "masked_multi_head_attention";
    }

    @Override
    public Class<MaskedMultiHeadAttention> targetClass() {
        return MaskedMultiHeadAttention.class;
    }

    @Override
    public void write(MaskedMultiHeadAttention layer, ObjectNode out) {
        out.put("head_count", layer.headCount());
        out.put("embedding_dim", layer.embeddingDim());
        out.put("attn_qkv_has_bias", layer.attnQkvHasBias());
        out.put("attn_out_has_bias", layer.attnOutHasBias());
    }

    @Override
    public MaskedMultiHeadAttention parse(JsonNode in) {
        int heads = in.get("head_count").asInt();
        int dim = in.get("embedding_dim").asInt();

        MaskedMultiHeadAttention mha = new MaskedMultiHeadAttention(new HardClipper(5), heads, dim);

        JsonNode qkv = in.get("attn_qkv_has_bias");
        JsonNode out = in.get("attn_out_has_bias");

        if (qkv != null) mha.attnQkvHasBias(qkv.asBoolean());
        if (out != null) mha.attnOutHasBias(out.asBoolean());

        return mha;
    }
}
