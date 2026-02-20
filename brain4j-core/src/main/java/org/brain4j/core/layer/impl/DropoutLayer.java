package org.brain4j.core.layer.impl;

import com.google.gson.JsonObject;
import org.brain4j.core.layer.Layer0;
import org.brain4j.math.Tensors;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Tensor;

import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * Implementation of a dropout layer, used to mitigate overfitting.
 * During training, it randomly turns to zero a fraction of the values in the input tensor.
 * During inference, the input doesn't change.
 *
 * @author xEcho1337
 */
public class DropoutLayer extends Layer0 {

    private final RandomGenerator random;
    private double dropoutRate;
    private int size;
    
    private DropoutLayer() {
        this.random = new SplittableRandom();
    }
    
    /**
     * Constructs a new dropout layer instance.
     * @param dropoutRate the dropout rate (0 < dropout < 1), specifying the probability of deactivating each neuron
     * @throws IllegalArgumentException if dropout is outside the range 0-1
     */
    public DropoutLayer(double dropoutRate) {
        if (dropoutRate < 0 || dropoutRate >= 1) {
            throw Commons.illegalArgument("Dropout must be greater or equal to 0 and less than 1!");
        }

        this.random = new SplittableRandom();
        this.dropoutRate = dropoutRate;
    }

    @Override
    public void connect() {
        this.size = previous.size();
    }

    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        if (!cache.isTraining()) return inputs;
        
        Tensor[] result = new Tensor[inputs.length];
        
        for (int i = 0; i < inputs.length; i++) {
            Tensor input = inputs[i];
            float[] mask = new float[input.elements()];
            
            for (int j = 0; j < mask.length; j++) {
                mask[j] = random.nextFloat() > dropoutRate ? 1 : 0;
            }
            
            Tensor tensorMask = Tensors.vector(mask);
            Tensor reshaped = input.reshapeGrad(input.elements());
            
            result[i] = reshaped.mulGrad(tensorMask)
                .div(1 - dropoutRate)
                .reshapeGrad(input.shape());
        }
        
        return result;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public void deserialize(JsonObject object) {
        this.dropoutRate = object.get("dropout").getAsDouble();
    }

    @Override
    public void serialize(JsonObject object) {
        object.addProperty("dropout", dropoutRate);
    }
    
    public RandomGenerator getRandom() {
        return random;
    }

    public double getDropoutRate() {
        return dropoutRate;
    }

    public DropoutLayer setDropoutRate(double dropoutRate) {
        this.dropoutRate = dropoutRate;
        return this;
    }

    public int getSize() {
        return size;
    }

    public DropoutLayer setSize(int size) {
        this.size = size;
        return this;
    }
}