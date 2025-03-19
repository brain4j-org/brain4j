package net.echo.brain4j.adapters.json;

import com.google.gson.*;
import net.echo.brain4j.training.updater.Updater;
import net.echo.brain4j.training.updater.impl.NormalUpdater;
import net.echo.brain4j.training.updater.impl.StochasticUpdater;

import java.lang.reflect.Type;

public class UpdaterAdapter implements JsonSerializer<Updater>, JsonDeserializer<Updater> {

    @Override
    public JsonElement serialize(Updater updater, Type type, JsonSerializationContext context) {
        JsonObject object = new JsonObject();

        object.addProperty("type", updater.getClass().getSimpleName());

        return object;
    }

    @Override
    public Updater deserialize(JsonElement jsonElement, Type type, JsonDeserializationContext context) throws JsonParseException {
        JsonObject object = jsonElement.getAsJsonObject();
        String optimizerType = object.get("type").getAsString();

        return switch (optimizerType) {
            case "StochasticUpdater" -> new StochasticUpdater();
            case "NormalUpdater" -> new NormalUpdater();
            default -> throw new IllegalArgumentException("Unknown updater type: " + optimizerType);
        };
    }
}
