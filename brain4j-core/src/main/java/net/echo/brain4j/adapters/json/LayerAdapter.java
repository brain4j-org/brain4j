package net.echo.brain4j.adapters.json;

import com.google.gson.*;
import net.echo.math.activation.Activations;
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
        object.addProperty("activation", layer.getActivation().getName());

        if (layer instanceof DenseLayer denseLayer) {
            object.addProperty("neurons", denseLayer.getTotalNeurons());
        }

        if (layer instanceof DropoutLayer dropoutLayer) {
            object.addProperty("rate", dropoutLayer.getDropout());
        }

        return object;
    }

    @Override
    public Layer deserialize(JsonElement element, Type type, JsonDeserializationContext context) throws JsonParseException {
        String layerType = element.getAsJsonObject().get("type").getAsString();
        String activationType = element.getAsJsonObject().get("activation").getAsString();

        Activations activation = Activations.valueOf(activationType);

        return switch (layerType) {
            case "DenseLayer" -> {
                int neurons = element.getAsJsonObject().get("neurons").getAsInt();
                yield new DenseLayer(neurons, activation);
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
