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
     * @param kernel the OpenCL kernel instance
     * @param input the current tensor
     * @param output the resulting tensor
     * @return a kernel factory ready to be launched
     */
    default SiliconKernel createKernel(
        ComputeFunction kernel,
        SiliconGpuTensor input,
        SiliconGpuTensor output
    ) {
        return SiliconKernel
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
        
        if (input instanceof SiliconGpuTensor gpuInput) {
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
        
        if (input instanceof SiliconGpuTensor gpuInput) {
            return computeGpu(gpuInput, "backward");
        }
        
        float[] inputData = input.data();
        float[] resultData = new float[inputData.length];

        for (int i = 0; i < resultData.length; i++) {
            resultData[i] = (float) derivative(inputData[i]);
        }

        Tensor derivative = Tensors.create(shape, resultData);
        return gradOut == null ? derivative : gradOut.times(derivative);
    }

    private SiliconGpuTensor computeGpu(SiliconGpuTensor input, String suffix) {
        SiliconDevice device = input.getDevice();
        SiliconGpuTensor result = new SiliconGpuTensor(device, input.shape());

        try (SiliconContext.QueueHandle queue = SiliconContext.getOrCreateQueue(device)) {
            System.out.println("Activating gpu kernel with id: " + getActivationId());
            ComputeFunction kernel = SiliconContext.findFunction(device, "activation_" + suffix);
            SiliconKernel factory = createKernel(kernel, input, result);

            ComputeSize size = new ComputeSize(input.size(), 1, 1);
            factory.launch(queue.queue(), size);
        }

        return result;
    }
}
