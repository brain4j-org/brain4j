package org.brain4j.core.layer.old.impl.convolutional;

import com.google.gson.JsonObject;
import org.brain4j.core.layer.old.OldLayer;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.activation.impl.Linear;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Tensor;

import java.util.Arrays;
import java.util.random.RandomGenerator;

public class ConvLayer extends OldLayer {

    private int channels;
    private int filters;
    private int kernelWidth;
    private int kernelHeight;
    private int stride = 1;
    private int padding = 0; // TODO: configurable
    
    private ConvLayer() {
    }
    
    public ConvLayer(int inputChannels, int filters, int kernelWidth, int kernelHeight) {
        this(inputChannels, filters, kernelWidth, kernelHeight, new Linear());
    }

    public ConvLayer(int inputChannels, int filters, int kernelWidth, int kernelHeight, int stride) {
        this(inputChannels, filters, kernelWidth, kernelHeight, stride, new Linear());
    }

    public ConvLayer(int inputChannels, int filters, int kernelWidth, int kernelHeight, Activations activation) {
        this(inputChannels, filters, kernelWidth, kernelHeight, activation.function());
    }

    public ConvLayer(int inputChannels, int filters, int kernelWidth,
                     int kernelHeight, int stride, Activations activation) {
        this(inputChannels, filters, kernelWidth, kernelHeight, stride, activation.function());
    }

    public ConvLayer(int inputChannels, int filters, int kernelWidth, int kernelHeight, Activation activation) {
        this(inputChannels, filters, kernelWidth, kernelHeight, 1, activation);
    }

    public ConvLayer(
        int inputChannels,
        int filters,
        int kernelWidth,
        int kernelHeight,
        int stride,
        Activation activation
    ) {
        this.channels = inputChannels;
        this.filters = filters;
        this.kernelWidth = kernelWidth;
        this.kernelHeight = kernelHeight;
        this.stride = stride;
        this.activation = activation;

        if (stride <= 0) {
            throw new IllegalArgumentException("Stride must be > 0. Got: " + stride);
        }
    }
    
    @Override
    public void connect() {
        this.bias = Tensors.zeros(filters).withGrad();
        this.weights = Tensors.zeros(filters, channels, kernelHeight, kernelWidth).withGrad();
    }
    
    @Override
    public void initWeights(RandomGenerator generator, int input, int output) {
        this.weights.map(x -> weightInit.generate(generator, input, output));
    }
    
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor input = inputs[0];
        cache.setStates(this, "input", input);
        
        validateInputTensor(input, "Input must have shape [N, C, H, W] while got: %s", Arrays.toString(input.shape()));

        Tensor convolved = input.convolveGrad(weights, stride);
        Tensor added = convolved.addGrad(bias.reshape(1, filters, 1, 1));
        
        cache.setStates(this, "pre_activation", added);

        return new Tensor[] { added.activateGrad(activation) };
    }
    
    @Override
    public void serialize(JsonObject object) {
        object.addProperty("channels", channels);
        object.addProperty("filters", filters);
        object.addProperty("kernel_width", kernelWidth);
        object.addProperty("kernel_height", kernelHeight);
        object.addProperty("stride", stride);
        object.addProperty("padding", padding);
    }
    
    @Override
    public void deserialize(JsonObject object) {
        this.channels = object.get("channels").getAsInt();
        this.filters = object.get("filters").getAsInt();
        this.kernelWidth = object.get("kernel_width").getAsInt();
        this.kernelHeight = object.get("kernel_height").getAsInt();
        this.stride = object.get("stride").getAsInt();
        this.padding = object.get("padding").getAsInt();
    }
    
    @Override
    public int size() {
        return filters;
    }
    
    @Override
    public boolean validInput(Tensor input) {
        // [batch, channels, height, width]
        return input.rank() == 4 && input.shapeAt(1) == channels;
    }
    
    public int getChannels() {
        return channels;
    }
    
    public void setChannels(int channels) {
        this.channels = channels;
    }
    
    public int getFilters() {
        return filters;
    }
    
    public void setFilters(int filters) {
        this.filters = filters;
    }
    
    public int getKernelWidth() {
        return kernelWidth;
    }
    
    public void setKernelWidth(int kernelWidth) {
        this.kernelWidth = kernelWidth;
    }
    
    public int getKernelHeight() {
        return kernelHeight;
    }
    
    public void setKernelHeight(int kernelHeight) {
        this.kernelHeight = kernelHeight;
    }
    
    public int getStride() {
        return stride;
    }
    
    public void setStride(int stride) {
        this.stride = stride;
    }
    
    public int getPadding() {
        return padding;
    }
    
    public void setPadding(int padding) {
        this.padding = padding;
    }
}
