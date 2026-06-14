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

public interface TransformerCodec<T extends Transformer<?>> extends Codec<T> {
    default void writeActivation(Activation activation, ObjectNode out) {
        Codec<Activation> codec = LayerIO.ACTIVATION_CODECS.get(activation.getClass());

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

    default  void writeNorm(Transformer.Config config, ObjectNode out) {
        Layer normInstance = config.normSupplier().get();
        Codec<? extends Layer> codec = LayerIO.LAYER_CODECS.get(normInstance.getClass());
        out.put("norm", codec.type());
    }

    default Activation parseActivation(JsonNode node) {
        if (node == null) {
            return new GELU();
        }

        if (node.isTextual()) {
            Codec<Activation> codec = LayerIO.ACTIVATION_CODECS.get(node.asText());

            if (codec == null) {
                throw Commons.illegalArgument("Unknown activation type: %s", node.asText());
            }

            return codec.parse(MAPPER.createObjectNode());
        }

        String type = node.get("type").asText();
        JsonNode config = node.has("config") ? node.get("config") : MAPPER.createObjectNode();
        Codec<Activation> codec = LayerIO.ACTIVATION_CODECS.get(type);

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
