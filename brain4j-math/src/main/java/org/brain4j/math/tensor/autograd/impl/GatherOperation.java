package org.brain4j.math.tensor.autograd.impl;

import org.brain4j.math.Tensors;
import org.brain4j.math.gpu.GpuContext;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.gpu.kernel.KernelFactory;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.autograd.Operation;
import org.brain4j.math.tensor.impl.GpuTensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class GatherOperation implements Operation {

    @Override
    public Tensor compute(Tensor... inputs) {
        Device device = resolveDevice(inputs);

        if (device == null) {
            return computeCpu(inputs[0], inputs[1]);
        }

        GpuTensor ids = toGpu(inputs[0], device);
        GpuTensor table = toGpu(inputs[1], device);

        int batchSize = ids.shape()[0];
        int seqLength = ids.shape()[1];
        int vocabSize = table.shape()[0];
        int embeddingDim = table.shape()[table.rank() - 1];

        GpuTensor output = new GpuTensor(device, new int[]{batchSize, seqLength, embeddingDim});

        try (GpuContext.QueueHandle qh = GpuContext.getOrCreateQueue(device)) {
            KernelFactory.create(device, "gather_forward")
                .buffer(ids.getDataBuffer())
                .buffer(table.getDataBuffer())
                .buffer(output.getDataBuffer())
                .intVal(batchSize * seqLength)
                .intVal(vocabSize)
                .intVal(embeddingDim)
                .launch(qh.queue(), Math.max(1, batchSize * seqLength * embeddingDim));
        }

        return output;
    }

    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor output, Tensor... inputs) {
        Device device = resolveDevice(gradOutput, inputs[0], inputs[1]);

        if (device == null) {
            return backwardCpu(gradOutput, inputs[0], inputs[1]);
        }

        GpuTensor ids = toGpu(inputs[0], device);
        GpuTensor table = toGpu(inputs[1], device);
        GpuTensor go = toGpu(gradOutput, device);

        int batchSize = ids.shape()[0];
        int seqLength = ids.shape()[1];
        int embeddingDim = table.shape()[table.rank() - 1];
        float[] idsData = ids.data();
        Map<Integer, List<Integer>> groups = new LinkedHashMap<>();

        for (int p = 0; p < batchSize * seqLength; p++) {
            int tokenId = (int) idsData[p];
            tokenId = Math.clamp(tokenId, 0, table.shape()[0] - 1);
            groups.computeIfAbsent(tokenId, k -> new ArrayList<>()).add(p);
        }

        int unique = groups.size();
        int[] uniqueIds = new int[unique];
        int[] rowPtr = new int[unique + 1];
        int[] positions = new int[batchSize * seqLength];

        int u = 0;
        int cursor = 0;

        for (Map.Entry<Integer, List<Integer>> entry : groups.entrySet()) {
            uniqueIds[u] = entry.getKey();
            rowPtr[u] = cursor;

            for (int pos : entry.getValue()) {
                positions[cursor++] = pos;
            }

            u++;
        }

        rowPtr[unique] = cursor;

        int[] tableShape = table.shape();
        GpuTensor gradTable = new GpuTensor(device, tableShape);

        try (GpuContext.QueueHandle qh = GpuContext.getOrCreateQueue(device)) {
            KernelFactory.create(device, "fill_const")
                .buffer(gradTable.getDataBuffer())
                .floatVal(0.0f)
                .intVal(gradTable.size())
                .launch(qh.queue(), Math.max(1, gradTable.size()));

            KernelFactory.create(device, "gather_backward_grouped")
                .buffer(go.getDataBuffer())
                .buffer(device.createBuffer(uniqueIds))
                .buffer(device.createBuffer(rowPtr))
                .buffer(device.createBuffer(positions))
                .buffer(gradTable.getDataBuffer())
                .intVal(unique)
                .intVal(embeddingDim)
                .launch(qh.queue(), Math.max(1, unique));
        }

        Tensor gradIds = Tensors.zeros(ids.shape()).to(device);
        return new Tensor[] { gradIds, gradTable };
    }

    private Tensor computeCpu(Tensor ids, Tensor table) {
        int[] idsShape = ids.shape();
        int batchSize = idsShape[0];
        int seqLength = idsShape[1];
        int embeddingDim = table.shape()[table.rank() - 1];

        float[] idsData = ids.data();
        float[] tableData = table.data();
        float[] outData = new float[batchSize * seqLength * embeddingDim];

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLength; s++) {
                int tokenId = (int) idsData[b * seqLength + s];
                int outOffset = (b * seqLength + s) * embeddingDim;
                int weightOffset = tokenId * embeddingDim;

                System.arraycopy(tableData, weightOffset, outData, outOffset, embeddingDim);
            }
        }

        return Tensors.create(new int[]{batchSize, seqLength, embeddingDim}, outData);
    }

    private Tensor[] backwardCpu(Tensor gradOutput, Tensor ids, Tensor table) {
        int[] tableShape = table.shape();
        int batchSize = ids.shape()[0];
        int seqLength = ids.shape()[1];
        int embeddingDim = tableShape[table.rank() - 1];

        float[] idsData = ids.data();
        float[] gradData = gradOutput.data();
        float[] tableGrad = new float[table.elements()];

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLength; s++) {
                int tokenId = (int) idsData[b * seqLength + s];
                int src = (b * seqLength + s) * embeddingDim;
                int dst = tokenId * embeddingDim;

                for (int d = 0; d < embeddingDim; d++) {
                    tableGrad[dst + d] += gradData[src + d];
                }
            }
        }

        Tensor gradTable = Tensors.create(Arrays.copyOf(tableShape, tableShape.length), tableGrad);
        return new Tensor[] { Tensors.zeros(ids.shape()), gradTable };
    }

    private Device resolveDevice(Tensor... tensors) {
        for (Tensor tensor : tensors) {
            if (tensor instanceof GpuTensor gpu) {
                return gpu.getDevice();
            }
        }

        return null;
    }

    private GpuTensor toGpu(Tensor tensor, Device device) {
        if (tensor instanceof GpuTensor gpu && gpu.getDevice() == device) {
            return gpu;
        }

        return (GpuTensor) tensor.to(device);
    }
}
