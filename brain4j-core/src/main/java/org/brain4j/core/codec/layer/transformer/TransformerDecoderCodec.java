package org.brain4j.core.codec.layer.transformer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.impl.transformer.Transformer;
import org.brain4j.math.activation.Activation;

import java.util.function.Supplier;

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
    public void write(Transformer.Decoder layer, ObjectNode out) {
        Transformer.Config config = layer.config();
        out.put("num_heads", config.heads());
        out.put("embedding_dim", config.embedDim());
        out.put("projection_dim", config.projDim());
        out.put("dropout", config.dropout());
        out.put("gating", config.gating());

        writeActivation(config.activation(), out);
        writeNorm(config, out);
    }

    @Override
    public Transformer.Decoder parse(JsonNode in) {
        int heads = in.get("num_heads").asInt();
        int dim = in.get("embedding_dim").asInt();
        double dropout = in.get("dropout").asDouble();
        int projDim = in.get("projection_dim").asInt();
        boolean gating = in.get("gating").asBoolean();

        Activation activation = parseActivation(in.get("activation"));
        Supplier<Layer> normSupplier = parseNormSupplier(in.get("norm"));

        Transformer.Config config = new Transformer.Config(dim, projDim, heads, dropout, gating, activation, normSupplier);
        return new Transformer.Decoder(config);
    }
}
