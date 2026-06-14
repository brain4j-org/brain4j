package org.brain4j.core.codec.layer.transformer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.Codec;
import org.brain4j.core.importing.io.LayerIO;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.impl.NormLayer;
import org.brain4j.core.layer.impl.RMSNormLayer;
import org.brain4j.core.layer.impl.transformer.Transformer;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.impl.GELU;
import org.brain4j.math.commons.Commons;

import java.util.function.Supplier;

import static org.brain4j.core.importing.format.impl.BrainFormat.MAPPER;

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
    public void write(Transformer.Encoder layer, ObjectNode out) {
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
    public Transformer.Encoder parse(JsonNode in) {
        int heads = in.get("num_heads").asInt();
        int dim = in.get("embedding_dim").asInt();
        double dropout = in.get("dropout").asDouble();
        int projDim = in.get("projection_dim").asInt();
        boolean gating = in.get("gating").asBoolean();

        Activation activation = parseActivation(in.get("activation"));
        Supplier<Layer> normSupplier = parseNormSupplier(in.get("norm"));

        Transformer.Config config = new Transformer.Config(dim, projDim, heads, dropout, gating, activation, normSupplier);
        return new Transformer.Encoder(config);
    }
}
