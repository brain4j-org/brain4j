package org.brain4j.core.layer.impl.convolutional;

import com.google.gson.JsonObject;
import org.brain4j.core.layer.Layer0;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.activation.impl.Linear;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

import java.util.Arrays;
import java.util.List;
import java.util.random.RandomGenerator;

public class ConvLayer extends Layer0 {

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
        Shape previousShape = previous.getOutputShapes().getFirst();
        
        this.bias = Tensors.zeros(filters).withGrad();
        this.weights = Tensors.zeros(filters, channels, kernelHeight, kernelWidth).withGrad();
        this.outputShape = List.of(inferOutputShape(previousShape));
    }
    
    @Override
    public void initWeights(RandomGenerator generator, int input, int output) {
        this.weights.map(x -> weightInit.generate(generator, input, output));
    }
    
    @Override
    public int getInputLength() {
        return 1;
    }
    
    public Shape inferOutputShape(Shape input) {
        if (input.rank() != 3) {
            throw new IllegalArgumentException(
                "ConvLayer expects input shape [C, H, W] but got: " + input
            );
        }
        
        int inChannels = input.dim(0);
        int inHeight = input.dim(1);
        int inWidth = input.dim(2);
        
        if (inChannels != channels) {
            throw new IllegalArgumentException(
                "Channel mismatch. Expected " + channels + " but got " + inChannels
            );
        }
        
        if (kernelHeight > inHeight + 2 * padding ||
            kernelWidth > inWidth + 2 * padding) {
            throw new IllegalArgumentException(
                "Kernel larger than input."
            );
        }
        
        int outHeight = (inHeight + 2 * padding - kernelHeight) / stride + 1;
        int outWidth  = (inWidth + 2 * padding - kernelWidth) / stride + 1;
        
        if ((inHeight + 2 * padding - kernelHeight) % stride != 0 ||
            (inWidth + 2 * padding - kernelWidth) % stride != 0) {
            throw new IllegalArgumentException(
                "Stride does not evenly divide spatial dimensions."
            );
        }
        
        return Shape.of(filters, outHeight, outWidth);
    }
    
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor input = inputs[0];
        cache.recordInput(this, input);
        
        validateInputTensor(input, "Input must have shape [N, C, H, W] while got: %s", Arrays.toString(input.shape()));

        Tensor convolved = input.convolveGrad(weights, stride);
        Tensor added = convolved.addGrad(bias.reshape(1, filters, 1, 1));
        
        cache.recordOutput(this, added);

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
