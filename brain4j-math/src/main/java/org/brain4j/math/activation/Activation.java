package org.brain4j.math.activation;

import org.brain4j.math.device.DeviceType;
import org.brain4j.math.kernel.KernelFactory;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;
import org.brain4j.math.tensor.gpu.OpenCLContext;
import org.brain4j.math.tensor.impl.CpuTensor;
import org.brain4j.math.tensor.impl.GpuTensor;
import org.brain4j.math.weights.WeightInitialization;
import org.jocl.cl_command_queue;
import org.jocl.cl_kernel;
import org.jocl.cl_program;

import static org.jocl.CL.clCreateKernel;

public abstract class Activation {

    protected cl_kernel forwardKernel;
    protected cl_kernel backwardKernel;

    /**
     * The default weight initialization for this activation function.
     * @return The default weight initialization.
     */
    public abstract WeightInitialization defaultWeightInit();

    /**
     * Activate a single scalar value (e.g. ReLU, Sigmoid, Tanh, etc.).
     */
    public abstract double activate(double input);

    /**
     * Get the derivative (scalar) of the activation at a single value.
     */
    public abstract double derivative(double input);

    public abstract String kernelPrefix();

    public KernelFactory createKernel(cl_kernel kernel, GpuTensor current, GpuTensor other) {
        return KernelFactory.create(kernel)
            .addMemParam(current.dataBuffer())
            .addMemParam(other.dataBuffer())
            .addIntParam(current.size());
    }

    public void initKernels(cl_program program) {
        this.forwardKernel = clCreateKernel(program, kernelPrefix() + "_forward", null);
        this.backwardKernel = clCreateKernel(program, kernelPrefix() + "_backward", null);
    }

    /**
     * Activate a vector of values.
     * Return a new vector containing the activated values.
     */
    public Tensor activate(Tensor input) {
        int[] shape = input.shape();

        if (input instanceof GpuTensor gpuInput) {
            GpuTensor result = new GpuTensor(gpuInput.shape());
            cl_command_queue queue = OpenCLContext.currentQueue();

            KernelFactory factory = createKernel(forwardKernel, gpuInput, result);
            factory.launch(queue, 1, gpuInput.size());

            return result;
        }

        float[] resultData = new float[input.elements()];
        float[] inputData = input.data();

        for (int i = 0; i < resultData.length; i++) {
            resultData[i] = (float) activate(inputData[i]);
        }

        return Tensors.create(shape, resultData);
    }

    /**
     * Get the derivative (vector) of the activation at a vector of values.
     */
    public Tensor derivative(Tensor input) {
        int[] shape = input.shape();

        if (input instanceof GpuTensor gpuInput) {
            GpuTensor result = new GpuTensor(gpuInput.shape());
            cl_command_queue queue = OpenCLContext.currentQueue();

            KernelFactory factory = createKernel(backwardKernel, gpuInput, result);
            factory.launch(queue, 1, gpuInput.size());

            return result;
        }

        float[] resultData = new float[input.elements()];
        float[] inputData = input.data();

        for (int i = 0; i < resultData.length; i++) {
            resultData[i] = (float) derivative(inputData[i]);
        }

        return Tensors.create(shape, resultData);
    }

    /**
     * Gets the default name for this activation function.
     * @return The name of the activation function.
     */
    public String name() {
        return getClass().getSimpleName().replaceAll("Activation", "");
    }
}
