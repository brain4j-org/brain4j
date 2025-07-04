package org.brain4j.core.clipper;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.common.tensor.impl.CpuTensor;
import org.brain4j.common.tensor.impl.GpuTensor;

/**
 * Gradient clipping is a technique used to prevent the gradients
 * from exploding or vanishing during the training process.
 */
public interface GradientClipper {

    default void clip(Tensor grad) {
       if (grad instanceof CpuTensor gradCpu) {
           clipCpu(gradCpu);
       }

       if (grad instanceof GpuTensor gradGpu) {
           clipGpu(gradGpu);
       }
    }

    /**
     * Clips the input gradient tensor in the CPU backend.
     * @param grad the gradient tensor
     */
    void clipCpu(CpuTensor grad);

    /**
     * Clips the input gradient tensor in the GPU backend.
     * @param grad the gradient tensor
     */
    void clipGpu(GpuTensor grad);

    /**
     * The name of the kernel of this gradient clip implementation.
     * @return the kernel name
     */
    String kernelName();
}