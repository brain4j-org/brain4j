package org.brain4j.core.layer.impl.utility;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import org.brain4j.core.layer.Layer;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Tensor;

/**
 * A utility layer that reshapes input tensors while preserving the batch dimension.
 * <p>
 * This layer applies a reshape operation to each input tensor, keeping the first
 * dimension unchanged (typically the batch size) and replacing the remaining
 * dimensions with the specified target shape.
 * <p>
 * The reshape operation is differentiable and preserves gradient flow.
 *
 * @apiNote this layer assumes that the first dimension of the input tensor
 *          represents the batch dimension and is always preserved
 *
 * @author xEcho1337
 */
public class ReshapeLayer extends Layer {

    private int[] shape;
    
    /**
     * DO NOT TOUCH: used for instancing when deserializing a model.
     */
    protected ReshapeLayer() {
    }
    
    /**
     * Creates a reshape layer with the specified target shape.
     *
     * @param shape the new shape applied after the batch dimension
     */
    public ReshapeLayer(int... shape) {
        this.shape = shape;
    }
    
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        checkInputLength(1, inputs);
        Tensor input = inputs[0];
        
        cache.rememberInput(this, input);

        int[] inputShape = input.shape();
        int[] newShape = new int[shape.length + 1];

        newShape[0] = inputShape[0];
        System.arraycopy(shape, 0, newShape, 1, shape.length);
        
        Tensor out = input.reshapeGrad(newShape);
        cache.rememberOutput(this, out);

        return new Tensor[] { out };
    }

    @Override
    public int size() {
        return shape[shape.length - 1];
    }
    
    @Override
    public void serialize(JsonObject object) {
        JsonArray array = new JsonArray();
        
        for (int dimension : shape) {
            array.add(dimension);
        }
        
        object.add("shape", array);
    }
    
    @Override
    public void deserialize(JsonObject object) {
        JsonArray data = object.getAsJsonArray("shape");
        this.shape = new int[data.size()];
        
        for (int i = 0; i < shape.length; i++) {
            shape[i] = data.get(i).getAsInt();
        }
    }
}
