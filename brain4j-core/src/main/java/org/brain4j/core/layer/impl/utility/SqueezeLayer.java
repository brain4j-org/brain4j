package org.brain4j.core.layer.impl.utility;

import com.google.gson.JsonObject;
import org.brain4j.core.layer.OldLayer;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Tensor;

public class SqueezeLayer extends OldLayer {
    
    private int dimension;
    private int size;

    protected SqueezeLayer() {
    }

    public SqueezeLayer(int dimension) {
        this.dimension = dimension;
    }

    @Override
    public void connect() {
        this.size = previous.size();
    }

    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor[] results = new Tensor[inputs.length];

        for (int i = 0; i < results.length; i++) {
            Tensor input = inputs[i];
            results[i] = dimension == -1 ? input.squeezeGrad() : input.squeezeGrad(dimension);
        }

        return results;
    }
    
    @Override
    public int size() {
        return size;
    }
    
    @Override
    public void serialize(JsonObject object) {
        object.addProperty("dimension", dimension);
    }
    
    @Override
    public void deserialize(JsonObject object) {
        this.dimension = object.get("dimension").getAsInt();
    }
}
