package org.brain4j.core.layer.impl;

import com.google.gson.JsonObject;
import org.brain4j.core.layer.OldLayer;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Tensor;

import java.util.random.RandomGenerator;

/**
 * Implementation of a fully connected (dense) neural network layer.
 * <p>
 * This layer performs a linear transformation on the input tensor,
 * followed by the application of a specified activation function.
 * </p>
 * <h2>Shape conventions:</h2>
 * <ul>
 *   <li>Input: {@code [batch, ..., input_dim]}</li>
 *   <li>Output: {@code [batch, ..., output_dim]}</li>
 *   <li>Weights: {@code [input_dim, output_dim]}</li>
 *   <li>Bias: {@code [output_dim]}</li>
 * </ul>
 * @implNote this layer supports multiple input tensors; assuming each one has the correct shape,
 *           each input tensor gets processed in the same way
 *
 * @author xEcho1337
 */
public class DenseLayer extends OldLayer {

    private int dimension;

    private DenseLayer() {
    }

    /**
     * Constructs a new instance of a dense layer with a linear activation.
     * @param dimension the dimension of the output
     */
    public DenseLayer(int dimension) {
        this(dimension, Activations.LINEAR);
    }

    /**
     * Constructs a new instance of a dense layer.
     * @param dimension the dimension of the output
     * @param activation the activation function
     */
    public DenseLayer(int dimension, Activations activation) {
        this.dimension = dimension;
        this.activation = activation.function();
    }

    /**
     * Constructs a new instance of a dense layer.
     * @param dimension the dimension of the output
     * @param activation the activation function
     */
    public DenseLayer(int dimension, Activation activation) {
        this.dimension = dimension;
        this.activation = activation;
    }

    @Override
    public void connect() {
        // Shape: [input_size, output_size]
        this.weights = Tensors.zeros(previous.size(), dimension).withGrad();
        this.bias = Tensors.zeros(dimension).withGrad();
    }

    @Override
    public void initWeights(RandomGenerator generator, int input, int output) {
        this.weights.map(x -> weightInit.generate(generator, input, output));
    }
    
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        validateInputLength(inputs);
        
        Tensor input = inputs[0];
        Tensor output = input.matmulGrad(weights);
        
        cache.setStates(this, "input", input);
        
        if (bias != null) output = output.addGrad(bias);

        Tensor beforeActivation = output;
        cache.setStates(this, "pre_activation", beforeActivation);
        
        return new Tensor[] { output.activateGrad(activation) };
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

    public int getDimension() {
        return dimension;
    }
}
