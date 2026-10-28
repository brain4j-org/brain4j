package org.brain4j.core.importing.io;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import org.brain4j.core.codec.JsonCodec;
import org.brain4j.core.codec.activation.ELUCodec;
import org.brain4j.core.codec.activation.GELUCodec;
import org.brain4j.core.codec.activation.LeakyReLUCodec;
import org.brain4j.core.codec.activation.LinearCodec;
import org.brain4j.core.codec.activation.MishCodec;
import org.brain4j.core.codec.activation.ReLUCodec;
import org.brain4j.core.codec.activation.SigmoidCodec;
import org.brain4j.core.codec.activation.SoftPlusCodec;
import org.brain4j.core.codec.activation.SoftmaxCodec;
import org.brain4j.core.codec.activation.SwishCodec;
import org.brain4j.core.codec.activation.TanhCodec;
import org.brain4j.core.codec.clipper.HardClipperCodec;
import org.brain4j.core.codec.clipper.L2ClipperCodec;
import org.brain4j.core.codec.clipper.NoClipperCodec;
import org.brain4j.core.codec.layer.ActivationCodec;
import org.brain4j.core.codec.layer.ConcatCodec;
import org.brain4j.core.codec.layer.ConvCodec;
import org.brain4j.core.codec.layer.DenseCodec;
import org.brain4j.core.codec.layer.DropoutCodec;
import org.brain4j.core.codec.layer.EmbeddingCodec;
import org.brain4j.core.codec.layer.LSTMCodec;
import org.brain4j.core.codec.layer.LiquidCodec;
import org.brain4j.core.codec.layer.MaskedMultiHeadAttentionCodec;
import org.brain4j.core.codec.layer.MaxPoolCodec;
import org.brain4j.core.codec.layer.MultiHeadAttentionCodec;
import org.brain4j.core.codec.layer.NormCodec;
import org.brain4j.core.codec.layer.OnnxOperationCodec;
import org.brain4j.core.codec.layer.PosEncodeCodec;
import org.brain4j.core.codec.layer.RMSNormCodec;
import org.brain4j.core.codec.layer.ReshapeCodec;
import org.brain4j.core.codec.layer.ScaleCodec;
import org.brain4j.core.codec.layer.SelectCodec;
import org.brain4j.core.codec.layer.SliceCodec;
import org.brain4j.core.codec.layer.SqueezeCodec;
import org.brain4j.core.codec.layer.transformer.TransformerDecoderCodec;
import org.brain4j.core.codec.layer.transformer.TransformerEncoderCodec;
import org.brain4j.core.codec.scaler.MinMaxScalerCodec;
import org.brain4j.core.codec.scaler.ZScoreScalerCodec;
import org.brain4j.core.codec.weightinit.LeCunInitCodec;
import org.brain4j.core.codec.weightinit.NormalHeInitCodec;
import org.brain4j.core.codec.weightinit.NormalInitCodec;
import org.brain4j.core.codec.weightinit.NormalXavierInitCodec;
import org.brain4j.core.codec.weightinit.UniformHeInitCodec;
import org.brain4j.core.codec.weightinit.UniformXavierInitCodec;
import org.brain4j.core.importing.Registry;
import org.brain4j.core.layer.Layer;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.clipper.GradientClipper;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.scaler.FeatureScaler;
import org.brain4j.math.weightsinit.WeightInit;

import static org.brain4j.core.importing.format.impl.BrainFormat.MAPPER;

/**
 * Reads and writes layers with JSON codecs.
 * Use it to serialize architectures for the native model format.
 */
public class LayerIO {

    public static final Registry<Layer> LAYER_CODECS = new Registry<>();
    public static final Registry<Activation> ACTIVATION_CODECS = new Registry<>();
    public static final Registry<WeightInit> WEIGHT_INIT_CODECS = new Registry<>();
    public static final Registry<GradientClipper> CLIPPER_CODECS = new Registry<>();
    public static final Registry<FeatureScaler> SCALER_CODECS = new Registry<>();

    static {
        LAYER_CODECS.put(
            new DenseCodec(), new LiquidCodec(), new NormCodec(), new ReshapeCodec(), new ConvCodec(), new LSTMCodec(),
            new DropoutCodec(), new MaxPoolCodec(), new RMSNormCodec(), new SqueezeCodec(), new SliceCodec(), new SelectCodec(),
            new ConcatCodec(), new EmbeddingCodec(), new PosEncodeCodec(), new MultiHeadAttentionCodec(),
            new MaskedMultiHeadAttentionCodec(), new TransformerEncoderCodec(), new TransformerDecoderCodec(),
            new ScaleCodec(), new ActivationCodec(), new OnnxOperationCodec()
        );

        ACTIVATION_CODECS.put(
            new ELUCodec(), new LeakyReLUCodec(), new SoftmaxCodec(), new GELUCodec(), new LinearCodec(),
            new MishCodec(), new ReLUCodec(), new SigmoidCodec(), new SoftPlusCodec(), new SwishCodec(), new TanhCodec()
        );

        WEIGHT_INIT_CODECS.put(
            new NormalInitCodec(), new NormalHeInitCodec(), new NormalXavierInitCodec(), new UniformHeInitCodec(),
            new UniformXavierInitCodec(), new LeCunInitCodec()
        );

        CLIPPER_CODECS.put(new NoClipperCodec(), new HardClipperCodec(), new L2ClipperCodec());
        SCALER_CODECS.put(new ZScoreScalerCodec(), new MinMaxScalerCodec());
    }

    /**
     * Parse a layer from its JSON node.
     *
     * @param node the node holding type, config and extras
     * @return the parsed layer
     */
    public static Layer parse(JsonNode node) {
        JsonNode config = node.get("config");

        if (config == null || !config.isObject()) {
            throw Commons.illegalArgument("Layer config must be an object");
        }

        String type = text(node.get("type"));
        JsonCodec<? extends Layer> codec = (JsonCodec<? extends Layer>) LAYER_CODECS.get(type);

        if (codec == null) {
            throw Commons.illegalArgument("Unknown layer type: %s", type);
        }

        Layer layer = codec.parse(config);

        JsonNode activationNode = node.get("activation");
        JsonNode weightInitNode = node.get("weight_init");
        JsonNode clipperNode = node.get("clipper");

        if (activationNode != null && !activationNode.isNull()) {
            layer.activation(parseActivation(activationNode));
        }

        if (weightInitNode != null && !weightInitNode.isNull()) {
            layer.weightInit(parseWeightInit(weightInitNode));
        }

        if (clipperNode != null && !clipperNode.isNull()) {
            layer.clipper(parseClipper(clipperNode));
        }

        return layer;
    }

    /**
     * Write a layer and its extras into the given container.
     *
     * @param container the target JSON object
     */
    public static void write(Layer layer, ObjectNode container) {
        JsonCodec<Layer> codec = layerCodec(layer);

        if (codec == null) {
            throw Commons.illegalArgument("Unexpected layer type: %s", layer.getClass().getName());
        }

        ObjectNode config = MAPPER.createObjectNode();
        codec.write(layer, config);

        container.put("type", codec.type());

        writeActivation(layer.activation(), container);
        writeWeightInit(layer.weightInit(), container);
        writeClipper(layer.clipper(), container);

        container.set("config", config);
    }

    /**
     * Return the stable serialized discriminator for a layer.
     *
     * @param layer layer to inspect
     * @return codec type
     */
    public static String type(Layer layer) {
        JsonCodec<Layer> codec = layerCodec(layer);

        if (codec == null) {
            throw Commons.illegalArgument("Unexpected layer type: %s", layer.getClass().getName());
        }

        return codec.type();
    }

    private static void writeActivation(Activation activation, ObjectNode container) {
        JsonCodec<Activation> codec = activationCodec(activation);

        if (codec == null) {
            throw Commons.illegalArgument("Unknown activation type: %s", activation.getClass().getName());
        }

        writeInfo("activation", container, codec, activation);
    }

    private static void writeWeightInit(WeightInit weightInit, ObjectNode container) {
        JsonCodec<WeightInit> codec = weightInitCodec(weightInit);

        if (codec == null) {
            throw Commons.illegalArgument("Unknown weight init type: %s", weightInit.getClass().getName());
        }

        writeInfo("weight_init", container, codec, weightInit);
    }

    private static void writeClipper(GradientClipper clipper, ObjectNode container) {
        JsonCodec<GradientClipper> codec = clipperCodec(clipper);

        if (codec == null) {
            throw Commons.illegalArgument("Unknown clipper type: %s", clipper.getClass().getName());
        }

        writeInfo("clipper", container, codec, clipper);
    }

    private static Activation parseActivation(JsonNode node) {
        if (node.isTextual()) {
            return parseCodec(ACTIVATION_CODECS, node.asText(), MAPPER.createObjectNode(), "activation");
        }

        String type = text(node.get("type"));
        JsonNode config = objectOrEmpty(node.get("config"));

        return parseCodec(ACTIVATION_CODECS, type, config, "activation");
    }

    private static WeightInit parseWeightInit(JsonNode node) {
        if (node.isTextual()) {
            return parseCodec(WEIGHT_INIT_CODECS, node.asText(), MAPPER.createObjectNode(), "weight init");
        }

        String type = text(node.get("type"));
        JsonNode config = objectOrEmpty(node.get("config"));

        return parseCodec(WEIGHT_INIT_CODECS, type, config, "weight init");
    }

    private static GradientClipper parseClipper(JsonNode node) {
        if (node.isTextual()) {
            return parseCodec(CLIPPER_CODECS, node.asText(), MAPPER.createObjectNode(), "clipper");
        }

        String type = text(node.get("type"));
        JsonNode config = objectOrEmpty(node.get("config"));

        return parseCodec(CLIPPER_CODECS, type, config, "clipper");
    }

    private static <T> T parseCodec(Registry<T> registry, String type, JsonNode config, String category) {
        JsonCodec<? extends T> codec = (JsonCodec<? extends T>) registry.get(type);

        if (codec == null) {
            throw Commons.illegalArgument("Unknown %s type: %s", category, type);
        }

        return codec.parse(config);
    }

    private static JsonNode objectOrEmpty(JsonNode node) {
        return node != null && node.isObject() ? node : MAPPER.createObjectNode();
    }

    private static String text(JsonNode node) {
        if (node == null || !node.isTextual()) {
            throw Commons.illegalArgument("Missing or invalid type field");
        }

        return node.asText();
    }

    private static <T> void writeInfo(String field, ObjectNode container, JsonCodec<T> codec, T value) {
        ObjectNode config = MAPPER.createObjectNode();
        codec.write(value, config);

        if (config.isEmpty()) {
            container.put(field, codec.type());
            return;
        }

        ObjectNode node = MAPPER.createObjectNode();
        node.put("type", codec.type());
        node.set("config", config);

        container.set(field, node);
    }

    private static JsonCodec<Layer> layerCodec(Layer layer) {
        return (JsonCodec<Layer>) LAYER_CODECS.get(layer.getClass());
    }

    private static JsonCodec<Activation> activationCodec(Activation activation) {
        return (JsonCodec<Activation>) ACTIVATION_CODECS.get(activation.getClass());
    }

    private static JsonCodec<WeightInit> weightInitCodec(WeightInit weightInit) {
        return (JsonCodec<WeightInit>) WEIGHT_INIT_CODECS.get(weightInit.getClass());
    }

    private static JsonCodec<GradientClipper> clipperCodec(GradientClipper clipper) {
        return (JsonCodec<GradientClipper>) CLIPPER_CODECS.get(clipper.getClass());
    }
}
