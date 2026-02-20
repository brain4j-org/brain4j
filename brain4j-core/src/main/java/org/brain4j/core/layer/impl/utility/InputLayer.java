package org.brain4j.core.layer.impl.utility;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import org.brain4j.core.layer.Layer0;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

import java.util.Arrays;
import java.util.List;

/**
 * Input layer used to define the expected shape of data entering the network.
 * <p>
 * This layer must always be placed as the first layer in a model.
 * It does not perform any computation but acts as a contract,
 * ensuring that tensors flowing into the model have the correct shape.
 * Only the last {@code N} dimensions are strictly checked, where
 * {@code N} is the rank of the specified input shape.
 * This allows an optional batch
 * dimension to be present without affecting validation.
 * </p>
 * <h2>Use example:</h2>
 * <blockquote><pre>{@code
 * ModelSpecs specs = ModelSpecs.of(
 *     // Expects input with shape [..., 3, 4]
 *     new InputLayer(3, 4),
 *     new DenseLayer(16),
 *     new DenseLayer(1)
 * );}
 * </pre></blockquote>
 *
 * @apiNote Individual dimensions may be set to {@code -1} to indicate that the
 *          corresponding size should not be checked.
 *
 * @author xEcho1337
 */
public class InputLayer extends Layer0 {

    private int[] shape;
    
    /**
     * DO NOT TOUCH: used for instancing when deserializing a model.
     */
    protected InputLayer() {
    }
    
    /**
     * Creates an input layer with the specified expected input shape.
     * @param shape the expected input shape, excluding any optional batch dimension
     */
    public InputLayer(int... shape) {
        this.shape = shape;
    }
    
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        for (Tensor input : inputs) {
            if (validInput(input)) continue;
            
            throw Commons.illegalArgument("Input must have shape %s! Got: %s",
                Arrays.toString(shape), Arrays.toString(input.shape()));
        }
        return inputs;
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
    
    @Override
    public boolean validInput(Tensor input) {
        if (input == null || input.rank() - 1 > shape.length) return false;

        int[] inputShape = input.shape();
        // give leniency, we only care about the trailing part of the shape
        int offset = inputShape.length - shape.length;
        
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] == -1) continue;
            if (inputShape[i + offset] != shape[i]) return false;
        }

        return true;
    }
    
    public int[] shape() {
        return shape;
    }
}
