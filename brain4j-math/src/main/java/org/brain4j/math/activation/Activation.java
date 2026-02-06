package org.brain4j.math.activation;

import org.brain4j.math.Tensors;
import org.brain4j.math.gpu.silicon.SiliconContext;
import org.brain4j.math.gpu.silicon.SiliconDevice;
import org.brain4j.math.gpu.silicon.SiliconKernel;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.SiliconGpuTensor;
import org.brain4j.math.weightsinit.WeightInit;
import org.silicon.api.function.ComputeFunction;
import org.silicon.api.kernel.ComputeSize;

import java.util.stream.IntStream;

public interface Activation {

    /**
     * The default weight initialization for this activation function.
     * @return The default weight initialization.
     */
    WeightInit defaultWeightInit();

    /**
     * Activate a single scalar value (e.g. ReLU, Sigmoid, Tanh, etc.).
     * @return the activation of the input
     */
    double activate(double input);

    /**
     * Gets the derivative (scalar) of the activation at a single value.
     * @return the derivative of the input as a double value
     */
    double derivative(double input);

    /**
     * Gets the kernel prefix for this activation
     * @return the prefix of the kernel
     */
    String getKernelPrefix();

    /**
     * Gets the default name for this activation function.
     * @return The name of the activation function.
     */
    default String name() {
        return getClass().getSimpleName().replaceAll("Activation", "");
    }

    /**
     * Creates the kernel to execute.
     * @param kernel the OpenCL kernel instance
     * @param current the current tensor
     * @param other the resulting tensor
     * @return a kernel factory ready to be launched
     */
    default SiliconKernel createKernel(ComputeFunction kernel, SiliconGpuTensor current, SiliconGpuTensor other) {
        return SiliconKernel
            .create(kernel)
            .buffer(current.getDataBuffer())
            .buffer(other.getDataBuffer())
            .intVal(current.size());
    }

    /**
     * Activate a vector of values.
     * Return a new vector containing the activated values.
     */
    default Tensor activate(Tensor input) {
        int[] shape = input.shape();
        
        if (input instanceof SiliconGpuTensor gpuInput) {
            SiliconDevice device = gpuInput.device();
            SiliconGpuTensor result = new SiliconGpuTensor(device, gpuInput.shape());
            
            try (SiliconContext.QueueHandle queue = SiliconContext.getOrCreateQueue(device)) {
                ComputeFunction kernel = SiliconContext.findFunction(device, getKernelPrefix() + "_forward");
                SiliconKernel factory = createKernel(kernel, gpuInput, result);
                
                ComputeSize size = new ComputeSize(gpuInput.size(), 1, 1);
                factory.launch(queue.queue(), size);
            }
            
            return result;
        }
        
        float[] inputData = input.data();
        float[] resultData = new float[inputData.length];
        
        if (resultData.length > 65536) {
            IntStream.range(0, inputData.length)
                .parallel()
                .forEach(i -> resultData[i] = (float) activate(inputData[i]));
        } else {
            for (int i = 0; i < resultData.length; i++) {
                resultData[i] = (float) activate(inputData[i]);
            }
        }

        return Tensors.create(shape, resultData);
    }

    /**
     * Get the derivative (vector) of the activation at a vector of values.
     */
    default Tensor derivative(Tensor input) {
        int[] shape = input.shape();
        
        if (input instanceof SiliconGpuTensor gpuInput) {
            SiliconDevice device = gpuInput.device();
            SiliconGpuTensor result = new SiliconGpuTensor(device, gpuInput.shape());
            
            try (SiliconContext.QueueHandle queue = SiliconContext.getOrCreateQueue(device)) {
                ComputeFunction kernel = SiliconContext.findFunction(device, getKernelPrefix() + "_backward");
                SiliconKernel factory = createKernel(kernel, gpuInput, result);
                
                ComputeSize size = new ComputeSize(gpuInput.size(), 1, 1);
                factory.launch(queue.queue(), size);
            }
            
            return result;
        }
        
        float[] inputData = input.data();
        float[] resultData = new float[inputData.length];
        
        if (resultData.length > 65536) {
            IntStream.range(0, inputData.length)
                .parallel()
                .forEach(i -> resultData[i] = (float) derivative(inputData[i]));
        } else {
            for (int i = 0; i < resultData.length; i++) {
                resultData[i] = (float) derivative(inputData[i]);
            }
        }

        return Tensors.create(shape, resultData);
    }
}
