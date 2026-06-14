package org.brain4j.math.activation;

import org.brain4j.math.Tensors;
import org.brain4j.math.gpu.GpuContext;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.gpu.kernel.KernelFactory;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.GpuTensor;
import org.brain4j.math.weightsinit.WeightInit;
import org.silicon.api.function.ComputeFunction;
import org.silicon.api.kernel.ComputeSize;

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
     * Gets the identifier of this activation
     * @return the activation identifier
     */
    int getActivationId();

    /**
     * Gets the default name for this activation function.
     * @return The name of the activation function.
     */
    default String name() {
        return getClass().getSimpleName().replaceAll("Activation", "");
    }

     /**
     * Creates the kernel to execute.
     * @param kernel the Silicon compute function
     * @param input the current tensor
     * @param output the resulting tensor
     * @return a kernel factory ready to be launched
     */
    default KernelFactory createKernel(
        ComputeFunction kernel,
        GpuTensor input,
        GpuTensor output
    ) {
        return KernelFactory
            .create(kernel)
            .intVal(getActivationId()) // activation type
            .floatVal(0f) // alpha
            .intVal(input.size()) // length
            .buffer(input.getDataBuffer())
            .buffer(output.getDataBuffer());
    }

    /**
     * Activate a vector of values.
     * Return a new vector containing the activated values.
     */
    default Tensor activate(Tensor input) {
        int[] shape = input.shape();
        
        if (input instanceof GpuTensor gpuInput) {
            return computeGpu(gpuInput, "forward");
        }
        
        float[] inputData = input.data();
        float[] resultData = new float[inputData.length];

        for (int i = 0; i < resultData.length; i++) {
            resultData[i] = (float) activate(inputData[i]);
        }

        return Tensors.create(shape, resultData);
    }

    /**
     * Get the derivative (vector) of the activation at a vector of values.
     */
    default Tensor derivative(Tensor input, Tensor output, Tensor gradOut) {
        int[] shape = input.shape();
        
        if (input instanceof GpuTensor gpuInput) {
            Tensor derivative = computeGpu(gpuInput, "backward");
            return gradOut == null ? derivative : derivative.mul(gradOut);
        }
        
        float[] inputData = input.data();
        float[] resultData = new float[inputData.length];

        for (int i = 0; i < resultData.length; i++) {
            resultData[i] = (float) derivative(inputData[i]);
        }

        Tensor derivative = Tensors.create(shape, resultData);
        return gradOut == null ? derivative : gradOut.times(derivative);
    }

    private GpuTensor computeGpu(GpuTensor input, String suffix) {
        Device device = input.getDevice();
        GpuTensor result = new GpuTensor(device, input.shape());

        try (GpuContext.QueueHandle queue = GpuContext.getOrCreateQueue(device)) {
            ComputeFunction kernel = GpuContext.findFunction(device, "activation_" + suffix);
            KernelFactory factory = createKernel(kernel, input, result);

            ComputeSize size = new ComputeSize(input.size(), 1, 1);
            factory.launch(queue.queue(), size);
        }

        return result;
    }
}
