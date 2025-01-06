package net.echo.brain4j.adapters;

import com.google.gson.*;
import net.echo.brain4j.training.optimizers.Optimizer;
import net.echo.brain4j.training.optimizers.impl.Adam;
import net.echo.brain4j.training.optimizers.impl.AdamW;
import net.echo.brain4j.training.optimizers.impl.GradientDescent;

import java.lang.reflect.Type;

public class OptimizerAdapter implements JsonSerializer<Optimizer>, JsonDeserializer<Optimizer> {

    @Override
    public JsonElement serialize(Optimizer optimizer, Type type, JsonSerializationContext context) {
        JsonObject object = new JsonObject();

        object.addProperty("type", optimizer.getClass().getSimpleName());

        JsonObject data = new JsonObject();

        data.addProperty("learningRate", optimizer.getLearningRate());

        if (optimizer instanceof Adam adam) {
            data.addProperty("beta1", adam.getBeta1());
            data.addProperty("beta2", adam.getBeta2());
            data.addProperty("epsilon", adam.getEpsilon());
        }

        if (optimizer instanceof AdamW adamW) {
            data.addProperty("weightDecay", adamW.getWeightDecay());
        }

        object.add("data", data);
        return object;
    }
    
    @Override
    public Optimizer deserialize(JsonElement jsonElement, Type type, JsonDeserializationContext context) throws JsonParseException {
        JsonObject object = jsonElement.getAsJsonObject();
        String optimizerType = object.get("type").getAsString();

        JsonObject data = object.get("data").getAsJsonObject();
        double learningRate = data.get("learningRate").getAsDouble();

        return switch (optimizerType) {
            case "SGD" -> new GradientDescent(learningRate);
            case "Adam" -> {
                Adam adam = new Adam(learningRate);

                adam.setBeta1(data.get("beta1").getAsDouble());
                adam.setBeta2(data.get("beta2").getAsDouble());
                adam.setEpsilon(data.get("epsilon").getAsDouble());

                yield adam;
            }
            case "AdamW" -> {
                AdamW adamW = new AdamW(learningRate);

                adamW.setBeta1(data.get("beta1").getAsDouble());
                adamW.setBeta2(data.get("beta2").getAsDouble());
                adamW.setEpsilon(data.get("epsilon").getAsDouble());
                adamW.setWeightDecay(data.get("weightDecay").getAsDouble());

                yield adamW;
            }
            default -> null;
        };
    }
}
