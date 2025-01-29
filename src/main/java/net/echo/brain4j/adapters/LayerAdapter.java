package net.echo.brain4j.adapters;

import com.google.gson.*;
import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.layer.impl.DropoutLayer;
import net.echo.brain4j.layer.impl.LayerNorm;

import java.lang.reflect.Type;

public class LayerAdapter implements JsonSerializer<Layer>, JsonDeserializer<Layer> {

    @Override
    public JsonElement serialize(Layer layer, Type type, JsonSerializationContext context) {
        JsonObject object = new JsonObject();

        object.addProperty("type", layer.getClass().getSimpleName());
        object.addProperty("activation", layer.getActivation().name());

        if (layer instanceof DenseLayer) {
            double[] biases = new double[layer.getNeurons().size()];

            for (int i = 0; i < biases.length; i++) {
                biases[i] = layer.getNeuronAt(i).getBias();
            }

            object.add("biases", context.serialize(biases));
        } else if (layer instanceof DropoutLayer dropoutLayer) {
            object.addProperty("rate", dropoutLayer.getDropout());
        }

        return object;
    }

    @Override
    public Layer deserialize(JsonElement element, Type type, JsonDeserializationContext context) throws JsonParseException {
        String layerType = element.getAsJsonObject().get("type").getAsString();
        String activationType = element.getAsJsonObject().get("activation").getAsString();

        Activations activations = Activations.valueOf(activationType);

        return switch (layerType) {
            case "DenseLayer" -> {
                double[] biases = context.deserialize(element.getAsJsonObject().get("biases"), double[].class);

                DenseLayer layer = new DenseLayer(biases.length, activations);

                for (int i = 0; i < layer.getNeurons().size(); i++) {
                    layer.getNeuronAt(i).setBias(biases[i]);
                }

                yield layer;
            }
            case "DropoutLayer" -> {
                double dropout = element.getAsJsonObject().get("rate").getAsDouble();
                yield new DropoutLayer(dropout);
            }
            case "LayerNorm" -> new LayerNorm();
            default -> throw new IllegalArgumentException("Unknown layer type: " + layerType);
        };
    }
}
