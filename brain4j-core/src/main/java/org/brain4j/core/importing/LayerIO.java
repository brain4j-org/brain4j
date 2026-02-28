package org.brain4j.core.importing;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.Codec;
import org.brain4j.core.codec.activation.*;
import org.brain4j.core.codec.clipper.HardClipperCodec;
import org.brain4j.core.codec.clipper.L2ClipperCodec;
import org.brain4j.core.codec.clipper.NoClipperCodec;
import org.brain4j.core.codec.layer.ConvCodec;
import org.brain4j.core.codec.layer.DenseCodec;
import org.brain4j.core.codec.layer.InputCodec;
import org.brain4j.core.codec.layer.NormCodec;
import org.brain4j.core.codec.layer.ReshapeCodec;
import org.brain4j.core.codec.weightinit.*;
import org.brain4j.core.layer.Layer;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.clipper.GradientClipper;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.weightsinit.WeightInit;

import static org.brain4j.core.importing.Registries.CLIPPERS_REGISTRY;
import static org.brain4j.core.importing.Registries.WEIGHT_INIT_REGISTRY;
import static org.brain4j.core.importing.format.impl.BrainFormat.MAPPER;

public class LayerIO {
    
    private static final Registry<Layer> LAYER_CODECS = new Registry<>();
    private static final Registry<Activation> ACTIVATION_CODECS = new Registry<>();
    private static final Registry<WeightInit> WEIGHT_INIT_CODECS = new Registry<>();
    private static final Registry<GradientClipper> CLIPPER_CODECS = new Registry<>();
    
    static {
        LAYER_CODECS.put(new InputCodec());
        LAYER_CODECS.put(new DenseCodec());
        LAYER_CODECS.put(new NormCodec());
        LAYER_CODECS.put(new ReshapeCodec());
        LAYER_CODECS.put(new ConvCodec());
        
        ACTIVATION_CODECS.put(new ELUCodec());
        ACTIVATION_CODECS.put(new LeakyReLUCodec());
        ACTIVATION_CODECS.put(new SoftmaxCodec());
        ACTIVATION_CODECS.put(new GELUCodec());
        ACTIVATION_CODECS.put(new LinearCodec());
        ACTIVATION_CODECS.put(new MishCodec());
        ACTIVATION_CODECS.put(new ReLUCodec());
        ACTIVATION_CODECS.put(new SigmoidCodec());
        ACTIVATION_CODECS.put(new SoftPlusCodec());
        ACTIVATION_CODECS.put(new SwishCodec());
        ACTIVATION_CODECS.put(new TanhCodec());
        
        WEIGHT_INIT_CODECS.put(new NormalInitCodec());
        WEIGHT_INIT_CODECS.put(new NormalHeInitCodec());
        WEIGHT_INIT_CODECS.put(new NormalXavierInitCodec());
        WEIGHT_INIT_CODECS.put(new UniformHeInitCodec());
        WEIGHT_INIT_CODECS.put(new UniformXavierInitCodec());
        WEIGHT_INIT_CODECS.put(new LeCunInitCodec());
        
        CLIPPER_CODECS.put(new NoClipperCodec());
        CLIPPER_CODECS.put(new HardClipperCodec());
        CLIPPER_CODECS.put(new L2ClipperCodec());
    }
    
    public static Layer parse(JsonNode node) {
        JsonNode config = node.get("config");
        
        if (config == null || !config.isObject()) {
            throw Commons.illegalArgument("Layer config must be an object");
        }
        
        String type = text(node.get("type"));
        Codec<? extends Layer> codec = LAYER_CODECS.get(type);
        
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
        Codec<Layer> codec = layerCodec(layer);
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
        Codec<Activation> codec = activationCodec(activation);

        if (codec == null) {
            throw Commons.illegalArgument("Unknown activation type: %s", activation.getClass().getName());
        }
        
        writeInfo("activation", container, codec, activation);
    }
    
    private static void writeWeightInit(WeightInit weightInit, ObjectNode container) {
        Codec<WeightInit> codec = weightInitCodec(weightInit);
        
        if (codec == null) {
            throw Commons.illegalArgument("Unknown weight init type: %s", weightInit.getClass().getName());
        }
        
        writeInfo("weight_init", container, codec, weightInit);
    }
    
    private static void writeClipper(GradientClipper clipper, ObjectNode container) {
        Codec<GradientClipper> codec = clipperCodec(clipper);
        
        if (codec == null) {
            throw Commons.illegalArgument("Unknown clipper type: %s", clipper.getClass().getName());
        }
        
        writeInfo("clipper", container, codec, clipper);
    }
    
    private static Activation parseActivation(JsonNode node) {
        if (node.isTextual()) {
            return Registries.ACTIVATION_REGISTRY.toInstance(node.asText());
        }
        
        String type = normalizeActivationType(text(node.get("type")));
        JsonNode config = objectOrEmpty(node.get("config"));
        
        Codec<? extends Activation> codec = ACTIVATION_CODECS.get(type);
        
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
        
        Codec<? extends WeightInit> codec = WEIGHT_INIT_CODECS.get(type);
        
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
        
        Codec<? extends GradientClipper> codec = CLIPPER_CODECS.get(type);
        
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
    
    private static String normalizeActivationType(String type) {
        return "leakyrelu".equals(type) ? "leaky_relu" : type;
    }
    
    private static <T> void writeInfo(String field, ObjectNode container, Codec<T> codec, T value) {
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
    
    @SuppressWarnings("unchecked")
    private static Codec<Layer> layerCodec(Layer layer) {
        return (Codec<Layer>) LAYER_CODECS.get((Class<? extends Layer>) layer.getClass());
    }
    
    @SuppressWarnings("unchecked")
    private static Codec<Activation> activationCodec(Activation activation) {
        return (Codec<Activation>) ACTIVATION_CODECS.get((Class<? extends Activation>) activation.getClass());
    }
    
    @SuppressWarnings("unchecked")
    private static Codec<WeightInit> weightInitCodec(WeightInit weightInit) {
        return (Codec<WeightInit>) WEIGHT_INIT_CODECS.get((Class<? extends WeightInit>) weightInit.getClass());
    }
    
    @SuppressWarnings("unchecked")
    private static Codec<GradientClipper> clipperCodec(GradientClipper clipper) {
        return (Codec<GradientClipper>) CLIPPER_CODECS.get((Class<? extends GradientClipper>) clipper.getClass());
    }
}
