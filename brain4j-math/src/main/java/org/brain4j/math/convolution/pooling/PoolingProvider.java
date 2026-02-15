package org.brain4j.math.convolution.pooling;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.SiliconGpuTensor;

public abstract class PoolingProvider {

    protected int stride;
    protected int windowHeight;
    protected int windowWidth;

    public PoolingProvider(int stride, int windowHeight, int windowWidth) {
        this.stride = stride;
        this.windowHeight = windowHeight;
        this.windowWidth = windowWidth;
    }
    
    public Tensor pool(Tensor input) {
        if (input instanceof SiliconGpuTensor gpuInput) {
            return poolGPU(gpuInput);
        } else return poolCPU(input);
    }

    public abstract Tensor poolCPU(Tensor input);
    
    public abstract Tensor poolGPU(Tensor input);
    
    public Tensor backward(Tensor gradient, Tensor input) {
        if (gradient instanceof SiliconGpuTensor gpuGrad
            && input instanceof SiliconGpuTensor gpuInput) {
            return backwardGPU(gpuGrad, gpuInput);
        } else return backwardCPU(gradient, input);
    }
    
    public abstract Tensor backwardCPU(Tensor gradient, Tensor input);
    
    public abstract Tensor backwardGPU(SiliconGpuTensor gradient, SiliconGpuTensor input);
    
}
