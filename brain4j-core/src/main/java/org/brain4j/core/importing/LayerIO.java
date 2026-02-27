package org.brain4j.core.importing;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.newimpl.*;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.impl.ELU;
import org.brain4j.math.activation.impl.LeakyReLU;
import org.brain4j.math.activation.impl.Softmax;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.weightsinit.WeightInit;

import static org.brain4j.core.importing.Registries.*;
import static org.brain4j.core.importing.format.impl.BrainFormat.MAPPER;

// Bigger TODO: Move the logic inside a proper Codec system
public class LayerIO {

    public static Layer parse(JsonNode node) {
        JsonNode config = node.get("config");
        JsonNode activation = config.get("activation");

        String type = node.get("type").asText();
        String weightInit = node.get("weight_init").asText();

        Layer layer = switch (type) {
            case "dense" -> {
                int dim = config.get("dimension").asInt();
                yield new DenseLayer(dim);
            }
            case "norm" -> {
                double epsilon = config.get("epsilon").asDouble();
                yield new NormLayer(epsilon);
            }
            case "input" -> {
                Shape shape = fromNode(config.get("shape"));
                yield new InputLayer(shape);
            }
            case "reshape" -> {
                Shape shape = fromNode(config.get("shape"));
                yield new ReshapeLayer(shape);
            }
            case "conv_2d" -> {
                int filters = config.get("filters").asInt();
                int kernelWidth = config.get("kernel_width").asInt();
                int kernelHeight = config.get("kernel_height").asInt();
                int stride = config.get("stride").asInt();

                yield new ConvLayer(filters, kernelWidth, kernelHeight, stride);
            }

            default -> throw new IllegalStateException("Unknown layer type: " + type);
        };

        Activation parsedActivation = parseActivation(activation);
        WeightInit parsedWeightInit = WEIGHT_INIT_REGISTRY.toInstance(weightInit);

        return layer
            .activation(parsedActivation)
            .weightInit(parsedWeightInit);
    }

    public static void write(Layer layer, ObjectNode container) {
        String clipper = CLIPPERS_REGISTRY.fromClass(layer.clipper().getClass());
        String weightInit = WEIGHT_INIT_REGISTRY.fromClass(layer.weightInit().getClass());

        ObjectNode config = MAPPER.createObjectNode();
        writeActivation(layer, container);

        // TODO: finish layers
        switch (layer) {
            case DenseLayer dense -> config.put("dimension", dense.outDimension());
            case NormLayer norm -> config.put("epsilon", norm.epsilon());
            case InputLayer input -> config.set("shape", toNode(input.shape()));
            case ReshapeLayer reshape -> config.set("shape", toNode(reshape.shape()));
            case ConvLayer conv -> {
                config.put("filters", conv.filters());
                config.put("channels", conv.channels());
                config.put("kernel_width", conv.kernelWidth());
                config.put("kernel_height", conv.kernelHeight());
                config.put("padding", conv.padding());
                config.put("stride", conv.stride());
            }
            default -> throw Commons.illegalArgument("Unexpected layer type: %s", layer);
        }

        container.put("clipper", clipper);
        container.put("weight_init", weightInit);
        container.set("config", config);
    }

    private static void writeActivation(Layer layer, ObjectNode container) {
        String activation = ACTIVATION_REGISTRY.fromClass(layer.activation().getClass());

        ObjectNode activationNode = MAPPER.createObjectNode();
        ObjectNode activationConfig = MAPPER.createObjectNode();

        switch (layer.activation()) {
            case LeakyReLU(double alpha) -> activationConfig.put("alpha", alpha);
            case ELU(double alpha) -> activationConfig.put("alpha", alpha);
            case Softmax(double temperature) -> activationConfig.put("temperature", temperature);
            default -> {}
        }

        activationNode.put("type", activation);
        activationNode.set("config", activationConfig);

        container.set("activation", activationNode);
    }

    private static Activation parseActivation(JsonNode activation) {
        String type = activation.get("type").asText();
        JsonNode config = activation.get("config");

        return switch (type) {
            case "leakyrelu" -> {
                double alpha = config.get("alpha").asDouble(0.01);
                yield new LeakyReLU(alpha);
            }
            case "elu" -> {
                double alpha = config.get("alpha").asDouble(1.0);
                yield new ELU(alpha);
            }
            case "softmax" -> {
                double temperature = config.get("temperature").asDouble(1.0);
                yield new Softmax(temperature);
            }
            default -> ACTIVATION_REGISTRY.toInstance(type);
        };
    }

    private static Shape fromNode(JsonNode node) {
        if (node == null || !node.isArray()) {
            throw Commons.illegalArgument("Shape must be an array");
        }

        int[] dims = new int[node.size()];

        for (int i = 0; i < node.size(); i++) {
            JsonNode dim = node.get(i);

            if (!dim.isInt()) {
                throw Commons.illegalArgument("Shape dimensions must be integers");
            }

            dims[i] = dim.intValue();
        }

        return Shape.of(dims);
    }

    private static ArrayNode toNode(Shape shape) {
        ArrayNode array = MAPPER.createArrayNode();
        for (int v : shape.dims()) {
            array.add(v);
        }
        return array;
    }
}
