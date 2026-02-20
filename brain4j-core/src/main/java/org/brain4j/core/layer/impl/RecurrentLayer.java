package org.brain4j.core.layer.impl;

import com.google.gson.JsonObject;
import org.brain4j.core.layer.Layer0;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.commons.Range;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.random.RandomGenerator;

public class RecurrentLayer extends Layer0 {

    private Tensor inputWeights;
    private Tensor hiddenWeights;
    private Tensor hiddenBias;
    private int dimension;
    private int hiddenDimension;
    
    private RecurrentLayer() {
    }
    
    /**
     * Constructs a new recurrent layer instance.
     *
     * @param dimension the dimension of the output
     * @param hiddenDimension the dimension of the hidden states
     * @param activation the activation function
     */
    public RecurrentLayer(int dimension, int hiddenDimension, Activations activation) {
        this.dimension = dimension;
        this.hiddenDimension = hiddenDimension;
        this.activation = activation.function();
        this.weightInit = this.activation.defaultWeightInit();
    }

    @Override
    public void connect() {
        int size = previous == null ? dimension : previous.size();
        this.inputWeights = Tensors.zeros(size, hiddenDimension).withGrad();
        this.hiddenWeights = Tensors.orthogonal(hiddenDimension, hiddenDimension).withGrad();
        this.hiddenBias = Tensors.zeros(hiddenDimension).withGrad();
        this.weights = Tensors.zeros(hiddenDimension, dimension).withGrad();
        this.bias = Tensors.zeros(dimension).withGrad();
    }

    @Override
    public void initWeights(RandomGenerator generator, int input, int output) {
        this.inputWeights.map(x -> weightInit.generate(generator, input, output));
        this.weights.map(x -> weightInit.generate(generator, hiddenDimension, output));
    }
    
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        // [batch, timesteps, dimension]
        Tensor input = inputs[0];

        validateInputTensor(input, "Input must have shape [batch, timesteps, dimension]! Got: %s", Arrays.toString(input.shape()));
        
        int batch = input.shapeAt(0);
        int timesteps = input.shapeAt(1);

        // [batch, timesteps, hidden_size]
        Tensor projectedInput = input.matmulGrad(inputWeights);
        Tensor hiddenState = Tensors.zeros(batch, hiddenDimension).withGrad();

        Tensor[] allStates = new Tensor[timesteps];

        for (int t = 0; t < timesteps; t++) {
            Range[] ranges = new Range[] { Range.all(), Range.point(t), Range.all() };

            Tensor timestepX = projectedInput.sliceGrad(ranges).squeezeGrad(1);
            Tensor timestepH = hiddenState.matmulGrad(hiddenWeights);

            hiddenState = timestepX.addGrad(timestepH).addGrad(hiddenBias).activateGrad(activation);
            allStates[t] = hiddenState.reshapeGrad(batch, 1, hiddenDimension);
        }
        
        // [batch, timesteps, hidden_dim]
        Tensor sequence = Tensors.concatGrad(List.of(allStates), 1);
        Tensor output = sequence.matmulGrad(weights).addGrad(bias);
        
        cache.recordOutput(this, output);
        return new Tensor[] { output };
    }
    
    @Override
    public void backward(StatesCache cache, Updater updater, Optimizer optimizer) {
        super.backward(cache, updater, optimizer);

        backward(inputWeights, updater, optimizer);
        backward(hiddenWeights, updater, optimizer);

        Tensor hiddenBiasGrad = hiddenBias.grad().sum(0, false);
        clipper.clip(hiddenBiasGrad);
        updater.change(hiddenBias, hiddenBiasGrad);
    }
    
    @Override
    public Layer0 freeze() {
        inputWeights.noGrad();
        hiddenWeights.noGrad();
        hiddenBias.noGrad();
        return super.freeze();
    }
    
    @Override
    public Layer0 unfreeze() {
        inputWeights.withGrad();
        hiddenWeights.withGrad();
        hiddenBias.withGrad();
        return super.unfreeze();
    }
    
    @Override
    public void serialize(JsonObject object) {
        object.addProperty("dimension", dimension);
        object.addProperty("hidden_dimension", hiddenDimension);
    }
    
    @Override
    public void deserialize(JsonObject object) {
        this.dimension = object.get("dimension").getAsInt();
        this.hiddenDimension = object.get("hidden_dimension").getAsInt();
    }
    
    @Override
    public void loadWeights(Map<String, Tensor> mappedWeights) {
        super.loadWeights(mappedWeights);
        this.inputWeights = mappedWeights.get("input_weights");
        this.hiddenWeights = mappedWeights.get("hidden_weights");
        this.hiddenBias = mappedWeights.get("hidden_bias");
    }
    
    @Override
    public boolean validInput(Tensor input) {
        return input.rank() == 3;
    }

    @Override
    public void resetGrad() {
        super.resetGrad();
        inputWeights.zeroGrad();
        hiddenWeights.zeroGrad();
        hiddenBias.zeroGrad();
    }
    
    @Override
    public int size() {
        return dimension;
    }
    
    @Override
    public int getTotalBias() {
        return hiddenBias.elements() + bias.elements();
    }
    
    @Override
    public int getTotalWeights() {
        return weights.elements() + inputWeights.elements() + hiddenWeights.elements();
    }
    
    @Override
    public Map<String, Tensor> weightsMap() {
        var result = super.weightsMap();
        result.put("input_weights", inputWeights);
        result.put("hidden_weights", hiddenWeights);
        result.put("hidden_bias", hiddenBias);
        return result;
    }
}
