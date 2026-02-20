package org.brain4j.core.layer.impl.convolutional;

import com.google.gson.JsonObject;
import org.brain4j.core.layer.Layer0;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Tensor;

public class MaxPoolLayer extends Layer0 {

    private int stride;
    private int windowHeight;
    private int windowWidth;
    private int size;

    private MaxPoolLayer() {
    }

    public MaxPoolLayer(int stride, int windowHeight, int windowWidth) {
        this.stride = stride;
        this.windowHeight = windowHeight;
        this.windowWidth = windowWidth;
    }

    @Override
    public void connect() {
        this.size = previous.size();
    }

    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        return new Tensor[] { inputs[0].maxPoolGrad(stride, windowHeight, windowWidth) };
    }

    @Override
    public void serialize(JsonObject object) {
        object.addProperty("stride", stride);
        object.addProperty("window_height", windowHeight);
        object.addProperty("window_width", windowWidth);
    }

    @Override
    public void deserialize(JsonObject object) {
        this.stride = object.getAsJsonPrimitive("stride").getAsInt();
        this.windowHeight = object.getAsJsonPrimitive("window_height").getAsInt();
        this.windowWidth = object.getAsJsonPrimitive("window_width").getAsInt();
    }

    @Override
    public int size() {
        return size;
    }
}
