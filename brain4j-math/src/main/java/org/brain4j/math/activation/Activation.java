package org.brain4j.math.activation;

import org.brain4j.math.device.DeviceType;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;
import org.brain4j.math.tensor.impl.CpuTensor;
import org.brain4j.math.weights.WeightInitialization;

public interface Activation {

    /**
     * The default weight initialization for this activation function.
     * @return The default weight initialization.
     */
    WeightInitialization defaultWeightInit();

    /**
     * Activate a single scalar value (e.g. ReLU, Sigmoid, Tanh, etc.).
     */
    double activate(double input);

    /**
     * Activate a vector of values.
     * Return a new vector containing the activated values.
     */
    default Tensor activate(Tensor input) {
        int[] shape = input.shape();

        DeviceType deviceType = input instanceof CpuTensor ? DeviceType.CPU : DeviceType.GPU;

        float[] resultData = new float[input.elements()];
        float[] inputData = input.data();

        for (int i = 0; i < resultData.length; i++) {
            resultData[i] = (float) activate(inputData[i]);
        }

        return Tensors.create(shape, resultData).to(deviceType);
    }

    /**
     * Get the derivative (scalar) of the activation at a single value.
     */
    double getDerivative(double input);

    /**
     * Get the derivative (vector) of the activation at a vector of values.
     */
    default Tensor getDerivative(Tensor input) {
        int[] shape = input.shape();

        DeviceType deviceType = input instanceof CpuTensor ? DeviceType.CPU : DeviceType.GPU;
        Tensor result = Tensors.zeros(shape);

        float[] resultData = result.data();
        float[] inputData = input.data();

        for (int i = 0; i < inputData.length; i++) {
            resultData[i] = (float) getDerivative(inputData[i]);
        }

        return result.to(deviceType);
    }

    /**
     * Gets the default name for this activation function.
     * @return The name of the activation function.
     */
    default String getName() {
        return getClass().getSimpleName().replaceAll("Activation", "");
    }
}
