package org.brain4j.core.layer.impl.utility;

import com.google.gson.JsonObject;
import org.brain4j.core.layer.Layer0;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Tensor;

/**
 * A utility layer that applies an element-wise activation function.
 * <p>
 * This layer does not introduce trainable parameters and simply transforms
 * its input tensors by applying the configured activation function. The output
 * shape is identical to the input shape.
 * <p>
 * The layer derives its dimensionality from the previous layer during the
 * {@link Layer0#connect()} phase.
 *
 * @author xEcho1337
 */
public class ActivationLayer extends Layer0 {

    private int dimension;
    
    /**
     * DO NOT TOUCH: used for instancing when deserializing a model.
     */
    protected ActivationLayer() {
    }
    
    /**
     * Creates an activation layer using a predefined activation function.
     *
     * @param activation the activation enum specifying the function to apply
     */
    public ActivationLayer(Activations activation) {
        this.activation = activation.function();
    }
    
    /**
     * Creates an activation layer with a custom activation function.
     *
     * @param activation the activation function to apply
     */
    public ActivationLayer(Activation activation) {
        this.activation = activation;
    }

    @Override
    public void connect() {
        this.dimension = previous.size();
    }
    
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor[] result = new Tensor[inputs.length];

        for (int i = 0; i < result.length; i++) {
            result[i] = inputs[i].activateGrad(activation);
        }

        cache.recordOutput(this, result);
        return result;
    }

    @Override
    public int size() {
        return dimension;
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
