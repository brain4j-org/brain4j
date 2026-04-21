package org.brain4j.core.layer.old;

import com.google.gson.JsonObject;
import org.brain4j.core.layer.Layer;
import org.brain4j.math.loss.LossFunction;
import org.brain4j.core.model.ModelBlock;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.impl.Linear;
import org.brain4j.math.clipper.GradientClipper;
import org.brain4j.math.clipper.impl.HardClipper;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.gpu.silicon.SiliconDevice;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.SiliconGpuTensor;
import org.brain4j.math.weightsinit.WeightInit;

import java.util.*;
import java.util.random.RandomGenerator;

/**
 * Abstract base class for all neural network layers.
 *
 * <p>A Layer is the fundamental building block of neural networks in Brain4J.
 * Each layer:
 * <ul>
 *   <li>Processes input tensors through forward/backward passes
 *   <li>Manages its own parameters (weights, biases)
 *   <li>Handles activation functions and gradient clipping
 *   <li>Can be serialized/deserialized for model saving
 * </ul>
 *
 * <p>Layers automatically handle both CPU and GPU execution through the tensor
 * abstraction, and support automatic differentiation for training.
 *
 * @author xEcho1337
 */
public abstract class OldLayer implements ModelBlock {

    protected Activation activation = new Linear();
    protected GradientClipper clipper = new HardClipper(5);
    protected WeightInit weightInit = activation.defaultWeightInit();
    protected OldLayer previous;
    protected Tensor weights;
    protected Tensor bias;
    protected boolean frozen;
    
    @Override
    public void appendTo(List<Layer> layers) {
        //layers.add(this);
    }
    
    public void connect(OldLayer previous) {
        this.previous = previous;
        connect();
    }
    
    /**
     * Constructs the tensors for weights in this layer.
     */
    public void connect() {
        // No-op
    }
    
    /**
     * Initializes the previously constructed weights with random values.
     * @param generator the random number generator
     * @param input the input dimension
     * @param output the output dimension
     */
    public void initWeights(RandomGenerator generator, int input, int output) {
        // No-op
    }
    
    /**
     * Performs a forward pass through this layer.
     *
     * @param cache the states cache for this forward pass
     * @param inputs the input tensors
     * @return the output tensors
     */
    public abstract Tensor[] forward(StatesCache cache, Tensor... inputs);
    
    public Tensor forward(StatesCache cache, Tensor input) {
        return forward(cache, new Tensor[] { input })[0];
    }

    protected void backward(Tensor tensor, Updater updater, Optimizer optimizer) {
        Tensor grad = tensor.grad();
        Tensor optimized = optimizer.step(weights, grad);

        clipper.clip(optimized);
        updater.change(weights, optimized);
    }

    /**
     * Computes the backward step for this layer, by calling the optimizer and scheduling weights update.
     *
     * @param cache the states cache of the forward pass
     * @param updater the updater of this model
     * @param optimizer the optimizer of this model
     */
    public void backward(StatesCache cache, Updater updater, Optimizer optimizer) {
        if (weights != null && weights.grad() != null) {
            backward(weights, updater, optimizer);
        }
        
        if (bias != null && bias.grad() != null) {
            Tensor biasGrad = bias.grad().sum(0, false);
            
            clipper.clip(biasGrad);
            updater.change(bias, biasGrad);
        }
    }
    
    /**
     * Computes the loss (the gradient) with respect to the loss function and launches the autograd.
     * This method should only be called for the last layer of the neural network.
     *
     * @param cache the state cache of this inference
     * @param labels the label tensors
     * @param outputs the output tensors
     * @param lossFunction the loss function of this model
     */
    public void computeLoss(
        StatesCache cache,
        Tensor[] labels,
        Tensor[] outputs,
        LossFunction lossFunction
    ) {
        Tensor[] preOutputs = cache.getStates(this, "pre_activation");
        
        if (labels.length != outputs.length) {
            throw Commons.illegalArgument("Labels amount does not match outputs amount!");
        }
        
        for (int i = 0; i < outputs.length; i++) {
            Tensor output = outputs[i];
            Tensor target = labels[i];
            Tensor preOutput = preOutputs[i];
            
            if (!Arrays.equals(output.shape(), target.shape())) {
                throw Commons.illegalState("Output and target shapes don't match! Output %s, Target: %s",
                    Arrays.toString(output.shape()), Arrays.toString(target.shape()));
            }

            Tensor derivative = activation.derivative(preOutput, output, null); // dy/dx
            Tensor delta = lossFunction.delta(output, target, derivative);
            
            preOutput.backward(delta);
        }
    }
    
    /**
     * Checks if the amount of inputs is greater than the maximum amount.
     * If so, throws an exception, otherwise will do nothing.
     * @param inputs the input tensors
     */
    public void validateInputLength(Tensor... inputs) {
        if (inputs.length != 1) {
            throw Commons.illegalArgument("Layer expects %s inputs but %s were given",
                inputs.length, 1);
        }
    }
    
    /**
     * Freezes all the trainable parameters in this layer.
     */
    public OldLayer freeze() {
        frozen = true;
        if (weights != null) weights.noGrad();
        if (bias != null) bias.noGrad();
        return this;
    }
    
    /**
     * Unfreezes all the parameters in this layer.
     */
    public OldLayer unfreeze() {
        frozen = false;
        if (weights != null) weights.withGrad();
        if (bias != null) bias.withGrad();
        return this;
    }
    
    public void serialize(JsonObject object) {
        // No-op
    }
    
    public void deserialize(JsonObject object) {
        // No-op
    }
    
    public void loadWeights(Map<String, Tensor> mappedWeights) {
        if (mappedWeights.containsKey("weights")) weights = mappedWeights.get("weights");
        if (mappedWeights.containsKey("bias")) bias = mappedWeights.get("bias");
    }
    
    /**
     * Returns the output size of this layer, i.e. the number of neurons.
     * This is useful for weights creation in consecutive layers.
     * @return the output size
     */
    public abstract int size();
    
    /**
     * Ports the weights of this layer to the specified device memory.
     * @param device the device to port the weights on
     */
    public void toDevice(SiliconDevice device) {
        if (weights != null) this.weights = toPersistentTensor(weights, device);
        if (bias != null) this.bias = toPersistentTensor(bias, device);
    }

    protected Tensor toPersistentTensor(Tensor tensor, SiliconDevice device) {
        if (tensor instanceof SiliconGpuTensor gpuTensor && gpuTensor.getDevice().equals(device)) {
            return gpuTensor;
        }
        
        Tensor result = device == null
            ? tensor.to(null)
            : SiliconGpuTensor.persistent(tensor, device);

        result.setAutogradContext(tensor.getAutogradContext());
        return result;
    }
    
    protected final void validateInputTensor(Tensor tensor, String message, Object... args) {
        if (validInput(tensor)) return;
        
        throw Commons.illegalArgument(message, args);
    }
    
    /**
     * Resets the gradients for all the weights in this layer.
     */
    public void resetGrad() {
        if (weights != null) weights.zeroGrad();
        if (bias != null) bias.zeroGrad();
    }

    /**
     * Validates if the input can be passed as an input to this layer.
     * This is done by checking the input dimension and comparing it
     * to the layer's expected dimension.
     *
     * @param input the input tensor
     * @return <code>true</code> if the input is valid, <code>false</code> otherwise
     */
    public boolean validInput(Tensor input) {
        return true;
    }
    
    /**
     * Gets the total number of biases in this layer.
     * @return 0 if bias is <code>null</code>, otherwise the number of elements in the bias tensor
     */
    public int getTotalBias() {
        if (bias == null) return 0;

        return bias.elements();
    }

    /**
     * Gets the total number of weights in this layer.
     * @return 0 if the weights is <code>null</code>, otherwise the number of elements in the weights tensor
     */
    public int getTotalWeights() {
        if (weights == null) return 0;

        return weights.elements();
    }
    
    public Map<String, Tensor> weightsMap() {
        Map<String, Tensor> result = new HashMap<>();
        
        if (weights != null) result.put("weights", weights);
        if (bias != null) result.put("bias", bias);
        
        return result;
    }
    
    @Override
    public OldLayer clone() {
        try {
            OldLayer clone = (OldLayer) super.clone();
            
            if (weights != null) {
                clone.weights = weights.copy();
                if (weights.usesGrad()) clone.weights.withGrad();
            }
            
            if (bias != null) {
                clone.bias = bias.copy();
                if (bias.usesGrad()) clone.bias.withGrad();
            }
            
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new AssertionError();
        }
    }

    public Activation getActivation() {
        return activation;
    }

    public void setActivation(Activation activation) {
        this.activation = activation;
    }

    public GradientClipper getClipper() {
        return clipper;
    }

    public void setClipper(GradientClipper clipper) {
        this.clipper = clipper;
    }

    public WeightInit getWeightInit() {
        return weightInit;
    }

    public void setWeightInit(WeightInit weightInit) {
        this.weightInit = weightInit;
    }

    public Tensor getWeights() {
        return weights;
    }

    public void setWeights(Tensor weights) {
        this.weights = weights;
    }

    public Tensor getBias() {
        return bias;
    }

    public void setBias(Tensor bias) {
        this.bias = bias;
    }

    public boolean isFrozen() {
        return frozen;
    }

    public void setFrozen(boolean frozen) {
        this.frozen = frozen;
    }
}
