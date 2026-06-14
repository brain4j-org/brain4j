package org.brain4j.math.tensor.autograd.impl;

import org.brain4j.math.Tensors;
import org.brain4j.math.commons.Range;
import org.brain4j.math.gpu.GpuContext;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.gpu.kernel.KernelFactory;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.autograd.Operation;
import org.brain4j.math.tensor.impl.GpuTensor;
import org.silicon.api.kernel.ComputeSize;

public record ConvolveOperation(int stride) implements Operation {

    public ConvolveOperation() {
        this(1);
    }

    @Override
    public Tensor compute(Tensor... inputs) {
        return inputs[0].convolve(inputs[1], stride);
    }

    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor output, Tensor... inputs) {
        Tensor input = inputs[0];
        Tensor filter = inputs[1];

        Device device = resolveDevice(input, filter, gradOutput);

        if (device == null) {
            return backwardCpu(gradOutput, input, filter);
        }
        
        return backwardGpu(
            input.to(device),
            filter.to(device),
            gradOutput.to(device),
            device
        );
    }

    private Tensor[] backwardGpu(Tensor input, Tensor filter, Tensor gradOutput, Device device) {
        if (!(input instanceof GpuTensor gpuInput)
            || !(filter instanceof GpuTensor gpuFilter)
            || !(gradOutput instanceof GpuTensor gpuGradOutput)) {
            throw new IllegalStateException("GPU backward expected GPU tensors");
        }

        int[] inputShape = gpuInput.shape();
        int[] filterShape = gpuFilter.shape();
        int[] gradOutputShape = gpuGradOutput.shape();

        if (inputShape.length != 4 || filterShape.length != 4 || gradOutputShape.length != 4) {
            throw new IllegalArgumentException("2D convolution backward requires 4D tensors");
        }

        int batch = inputShape[0];
        int inChannels = inputShape[1];
        int inHeight = inputShape[2];
        int inWidth = inputShape[3];

        int numFilters = filterShape[0];
        int kernelHeight = filterShape[2];
        int kernelWidth = filterShape[3];

        int outHeight = gradOutputShape[2];
        int outWidth = gradOutputShape[3];

        GpuTensor gradInput = new GpuTensor(device, inputShape);
        GpuTensor gradFilter = new GpuTensor(device, filterShape);

        int tile = 8;
        ComputeSize local = new ComputeSize(tile, tile, 1);

        ComputeSize gradInputGlobal = new ComputeSize(
            roundUp(inWidth, tile),
            roundUp(inHeight * inChannels, tile),
            Math.max(1, batch)
        );

        ComputeSize gradFilterGlobal = new ComputeSize(
            roundUp(kernelWidth, tile),
            roundUp(kernelHeight * inChannels, tile),
            Math.max(1, numFilters)
        );

        try (GpuContext.QueueHandle qh = GpuContext.getOrCreateQueue(device)) {
            KernelFactory.create(device, "conv2d_backward_input_nchw")
                .buffer(gpuGradOutput.getDataBuffer())
                .buffer(gpuFilter.getDataBuffer())
                .buffer(gradInput.getDataBuffer())
                .intVal(batch)
                .intVal(inChannels)
                .intVal(inHeight)
                .intVal(inWidth)
                .intVal(numFilters)
                .intVal(kernelHeight)
                .intVal(kernelWidth)
                .intVal(stride)
                .intVal(outHeight)
                .intVal(outWidth)
                .launch(qh.queue(), gradInputGlobal, local);

            KernelFactory.create(device, "conv2d_backward_filter_nchw")
                .buffer(gpuInput.getDataBuffer())
                .buffer(gpuGradOutput.getDataBuffer())
                .buffer(gradFilter.getDataBuffer())
                .intVal(batch)
                .intVal(inChannels)
                .intVal(inHeight)
                .intVal(inWidth)
                .intVal(numFilters)
                .intVal(kernelHeight)
                .intVal(kernelWidth)
                .intVal(stride)
                .intVal(outHeight)
                .intVal(outWidth)
                .launch(qh.queue(), gradFilterGlobal, local);
        }

        return new Tensor[] { gradInput, gradFilter };
    }

    private Tensor[] backwardCpu(Tensor gradOutput, Tensor input, Tensor filter) {
        int[] inputShape = input.shape();
        int[] filterShape = filter.shape();

        int batch = inputShape[0];
        int inChannels = inputShape[1];
        int inHeight = inputShape[2];
        int inWidth = inputShape[3];

        int numFilters = filterShape[0];
        int filterHeight = filterShape[2];
        int filterWidth = filterShape[3];

        int outHeight = gradOutput.shape()[2];
        int outWidth = gradOutput.shape()[3];

        Tensor gradInput = Tensors.zerosLike(input);
        Tensor gradFilter = Tensors.zerosLike(filter);

        int batchSize = inChannels * inHeight * inWidth;
        float[] gradInputData = gradInput.data();

        for (int b = 0; b < batch; b++) {
            Tensor inputBatch = input.slice(Range.point(b)).squeeze(0);
            Tensor dOutBatch = gradOutput.slice(Range.point(b));

            Tensor xCol = Tensors.im2col(inputBatch, filterHeight, filterWidth, stride);
            Tensor wCol = filter.reshape(numFilters, inChannels * filterHeight * filterWidth);

            Tensor dYCol = dOutBatch.reshape(numFilters, outHeight * outWidth);
            Tensor dWCol = dYCol.matmul(xCol.transpose());
            gradFilter.add(dWCol.reshape(filter.shape()));

            Tensor dXCol = wCol.transpose().matmul(dYCol);
            Tensor dInputBatch = Tensors.col2im(
                dXCol, inChannels, inHeight, inWidth, filterHeight, filterWidth, stride
            );

            float[] dInputData = dInputBatch.data();
            System.arraycopy(dInputData, 0, gradInputData, b * batchSize, batchSize);
        }

        return new Tensor[] { gradInput, gradFilter };
    }

    private Device resolveDevice(Tensor... tensors) {
        for (Tensor tensor : tensors) {
            if (tensor instanceof GpuTensor gpu) {
                return gpu.getDevice();
            }
        }
        return null;
    }

    private int roundUp(int value, int tile) {
        int remainder = value % tile;
        return remainder == 0 ? value : value + tile - remainder;
    }
}
