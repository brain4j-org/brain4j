package org.brain4j.core.codec.layer.transformer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.JsonCodec;
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

public interface TransformerCodec<T extends Transformer<?>> extends JsonCodec<T> {

    @Override
    default void write(T value, ObjectNode out) {
        Transformer.Config config = value.config();
        out.put("num_heads", config.heads());
        out.put("embedding_dim", config.embedDim());
        out.put("projection_dim", config.projDim());
        out.put("dropout", config.dropout());
        out.put("gating", config.gating());

        writeActivation(config.activation(), out);
        writeNorm(config, out);
    }

    default Transformer.Config readConfig(JsonNode in) {
        int heads = in.get("num_heads").asInt();
        int dim = in.get("embedding_dim").asInt();
        double dropout = in.get("dropout").asDouble();
        int projDim = in.get("projection_dim").asInt();
        boolean gating = in.get("gating").asBoolean();

        Activation activation = parseActivation(in.get("activation"));
        Supplier<Layer> normSupplier = parseNormSupplier(in.get("norm"));

        return new Transformer.Config(dim, projDim, heads, dropout, gating, activation, normSupplier);
    }

    default void writeActivation(Activation activation, ObjectNode out) {
        JsonCodec<Activation> codec = (JsonCodec<Activation>) LayerIO.ACTIVATION_CODECS.get(activation.getClass());

        if (codec == null) {
            throw Commons.illegalArgument("Unknown activation type: %s", activation.getClass().getName());
        }

        ObjectNode config = MAPPER.createObjectNode();
        codec.write(activation, config);

        if (config.isEmpty()) {
            out.put("activation", codec.type());
            return;
        }

        ObjectNode node = MAPPER.createObjectNode();
        node.put("type", codec.type());
        node.set("config", config);
        out.set("activation", node);
    }

    default void writeNorm(Transformer.Config config, ObjectNode out) {
        Layer normInstance = config.normSupplier().get();
        JsonCodec<? extends Layer> codec = (JsonCodec<? extends Layer>) LayerIO.LAYER_CODECS.get(normInstance.getClass());
        out.put("norm", codec.type());
    }

    default Activation parseActivation(JsonNode node) {
        if (node == null) {
            return new GELU();
        }

        if (node.isTextual()) {
            JsonCodec<Activation> codec = (JsonCodec<Activation>) LayerIO.ACTIVATION_CODECS.get(node.asText());

            if (codec == null) {
                throw Commons.illegalArgument("Unknown activation type: %s", node.asText());
            }

            return codec.parse(MAPPER.createObjectNode());
        }

        String type = node.get("type").asText();
        JsonNode config = node.has("config") ? node.get("config") : MAPPER.createObjectNode();
        JsonCodec<Activation> codec = (JsonCodec<Activation>) LayerIO.ACTIVATION_CODECS.get(type);

        if (codec == null) {
            throw Commons.illegalArgument("Unknown activation type: %s", type);
        }

        return codec.parse(config);
    }

    default Supplier<Layer> parseNormSupplier(JsonNode node) {
        if (node == null) {
            return NormLayer::new;
        }

        String type = node.asText();

        return switch (type) {
            case "norm" -> NormLayer::new;
            case "rms_norm" -> RMSNormLayer::new;
            default -> throw Commons.illegalArgument("Unknown norm type: %s", type);
        };
    }
}
