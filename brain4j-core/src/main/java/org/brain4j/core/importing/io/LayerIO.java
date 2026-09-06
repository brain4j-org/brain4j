package org.brain4j.core.importing.io;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.JsonCodec;
import org.brain4j.core.codec.activation.*;
import org.brain4j.core.codec.clipper.HardClipperCodec;
import org.brain4j.core.codec.clipper.L2ClipperCodec;
import org.brain4j.core.codec.clipper.NoClipperCodec;
import org.brain4j.core.codec.layer.*;
import org.brain4j.core.codec.layer.transformer.TransformerDecoderCodec;
import org.brain4j.core.codec.layer.transformer.TransformerEncoderCodec;
import org.brain4j.core.codec.scaler.MinMaxScalerCodec;
import org.brain4j.core.codec.scaler.ZScoreScalerCodec;
import org.brain4j.core.codec.weightinit.*;
import org.brain4j.core.importing.Registries;
import org.brain4j.core.importing.Registry;
import org.brain4j.core.layer.Layer;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.clipper.GradientClipper;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.scaler.FeatureScaler;
import org.brain4j.math.weightsinit.WeightInit;

import static org.brain4j.core.importing.Registries.CLIPPERS_REGISTRY;
import static org.brain4j.core.importing.Registries.WEIGHT_INIT_REGISTRY;
import static org.brain4j.core.importing.format.impl.BrainFormat.MAPPER;

public class LayerIO {

    public static final Registry<Layer> LAYER_CODECS = new Registry<>();
    public static final Registry<Activation> ACTIVATION_CODECS = new Registry<>();
    public static final Registry<WeightInit> WEIGHT_INIT_CODECS = new Registry<>();
    public static final Registry<GradientClipper> CLIPPER_CODECS = new Registry<>();
    public static final Registry<FeatureScaler> SCALER_CODECS = new Registry<>();
    
    static {
        LAYER_CODECS.put(
            new InputCodec(), new DenseCodec(), new LiquidCodec(), new NormCodec(), new ReshapeCodec(), new ConvCodec(), new LSTMCodec(),
            new DropoutCodec(), new MaxPoolCodec(), new RMSNormCodec(), new SqueezeCodec(), new SliceCodec(), new SelectCodec(),
            new ConcatCodec(), new EmbeddingCodec(), new PosEncodeCodec(), new MultiHeadAttentionCodec(),
            new MaskedMultiHeadAttentionCodec(), new TransformerEncoderCodec(), new TransformerDecoderCodec(),
            new ScaleCodec(), new ActivationCodec(), new org.brain4j.core.codec.layer.OnnxOperationCodec()
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

    public static void write(Layer layer, ObjectNode container) {
        JsonCodec<Layer> codec = (JsonCodec<Layer>) layerCodec(layer);
        if (codec == null) {
            throw Commons.illegalArgument("Unexpected layer type: %s", layer.getClass().getName());
        }

        ObjectNode config = MAPPER.createObjectNode();
        codec.write(layer, config);

        writeActivation(layer.activation(), container);
        writeWeightInit(layer.weightInit(), container);
        writeClipper(layer.clipper(), container);

        container.set("config", config);
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
            return Registries.ACTIVATION_REGISTRY.toInstance(node.asText());
        }

        String type = text(node.get("type"));
        JsonNode config = objectOrEmpty(node.get("config"));

        JsonCodec<? extends Activation> codec = (JsonCodec<? extends Activation>) ACTIVATION_CODECS.get(type);

        if (codec != null) {
            return codec.parse(config);
        }

        return Registries.ACTIVATION_REGISTRY.toInstance(type);
    }

    private static WeightInit parseWeightInit(JsonNode node) {
        if (node.isTextual()) {
            return WEIGHT_INIT_REGISTRY.toInstance(node.asText());
        }

        String type = text(node.get("type"));
        JsonNode config = objectOrEmpty(node.get("config"));

        JsonCodec<? extends WeightInit> codec = (JsonCodec<? extends WeightInit>) WEIGHT_INIT_CODECS.get(type);

        if (codec != null) {
            return codec.parse(config);
        }

        return WEIGHT_INIT_REGISTRY.toInstance(type);
    }

    private static GradientClipper parseClipper(JsonNode node) {
        if (node.isTextual()) {
            return CLIPPERS_REGISTRY.toInstance(node.asText());
        }

        String type = text(node.get("type"));
        JsonNode config = objectOrEmpty(node.get("config"));

        JsonCodec<? extends GradientClipper> codec = (JsonCodec<? extends GradientClipper>) CLIPPER_CODECS.get(type);

        if (codec != null) {
            return codec.parse(config);
        }

        return CLIPPERS_REGISTRY.toInstance(type);
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
