package org.brain4j.core.layer.newimpl;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.impl.Linear;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

public class ConvLayer extends Layer {
    
    private final int filters;
    private final int kernelWidth;
    private final int kernelHeight;
    private final int padding = 0; // TODO: configurable
    private final int stride;
    
    public ConvLayer(int filters, int kernelWidth, int kernelHeight) {
        this(filters, kernelWidth, kernelHeight, new Linear());
    }
    
    public ConvLayer(int filters, int kernelWidth, int kernelHeight, int stride) {
        this(filters, kernelWidth, kernelHeight, stride, new Linear());
    }
    
    public ConvLayer(int filters, int kernelWidth, int kernelHeight, Activation activation) {
        this(filters, kernelWidth, kernelHeight, 1, activation);
    }
    
    public ConvLayer(int filters, int kernelWidth, int kernelHeight, int stride, Activation activation) {
        super(activation);
        this.filters = filters;
        this.kernelWidth = kernelWidth;
        this.kernelHeight = kernelHeight;
        this.stride = stride;
        
        if (stride <= 0) throw Commons.illegalArgument("Stride must be > 0. Got: %s", stride);
        if (filters <= 0) throw Commons.illegalArgument("Filters must be > 0 Got: %s", filters);
        if (kernelWidth <= 0) throw Commons.illegalArgument("Kernel width must be > 0 Got: %s", kernelWidth);
        if (kernelHeight <= 0) throw Commons.illegalArgument("Kernel height must be > 0 Got: %s", kernelHeight);
    }
    
    @Override
    public void build(List<Shape> inputShapes) {
        Shape inputShape = inputShapes.getFirst();
        int channels = inputShape.dim(0);
        
        Tensor kernel = Tensors.zeros(filters, channels, kernelHeight, kernelWidth);
        Tensor bias = Tensors.zeros(filters);
        
        parameters.put("kernel", kernel);
        parameters.put("bias", bias);
    }
    
    @Override
    public void initWeights(List<Shape> inputShapes, RandomGenerator rng) {
        Shape inputShape = inputShapes.getFirst();
        int channels = inputShape.dim(0);
        
        int input = channels * kernelHeight * kernelWidth;
        int output = filters * kernelHeight * kernelWidth;
        
        generateWeightsFor("kernel", rng, input, output);
    }
    
    @Override
    public List<Shape> inferOutputShapes(List<Shape> inputShapes) {
        if (inputShapes.size() != 1) {
            throw Commons.illegalArgument("Layer requires 1 input but %s were given!", inputShapes.size());
        }
        
        Shape inputShape = inputShapes.getFirst();
        
        if (inputShape.rank() != 3) {
            throw Commons.illegalArgument("ConvLayer requires tensors with rank 3 but %s were given!", inputShape.rank());
        }
        
        int height = inputShape.dim(1);
        int width = inputShape.dim(2);
        
        int numeratorH = height - kernelHeight + 2 * padding;
        int numeratorW = width - kernelWidth + 2 * padding;
        
        if (numeratorH < 0 || numeratorW < 0) {
            throw Commons.illegalArgument("Kernel is too big for input!.");
        }
        
        int outHeight = numeratorH / stride + 1;
        int outWidth  = numeratorW / stride + 1;
        
        if (outHeight <= 0 || outWidth <= 0) {
            throw Commons.illegalArgument("Negative output dims: outHeight=%s outWidth=%s", outHeight, outWidth);
        }
        
        return List.of(Shape.of(filters, outHeight, outWidth));
    }
    
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor input = inputs[0];
        
        if (input.rank() != 4) {
            throw Commons.illegalArgument("Expected input with rank 4 but got %s", input.rank());
        }
        
        Tensor W = getParam("kernel");
        Tensor B = getParam("bias");
        
        Tensor result = input.convolveGrad(W, stride)
            .addGrad(B.reshape(1, filters, 1, 1))
            .activateGrad(activation);
        
        return tensors(result);
    }
    
    @Override
    public Layer copy() {
        ConvLayer copy = new ConvLayer(filters, kernelWidth, kernelHeight, stride, activation);
        copyParameters(copy);
        return copy;
    }
}
