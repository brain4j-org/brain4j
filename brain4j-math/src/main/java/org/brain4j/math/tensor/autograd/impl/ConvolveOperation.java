package org.brain4j.math.tensor.autograd.impl;

import org.brain4j.math.Tensors;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.autograd.Operation;
import org.brain4j.math.commons.Range;

import java.util.stream.IntStream;

public record ConvolveOperation(int stride) implements Operation {

    public ConvolveOperation() {
        this(1);
    }

    @Override
    public Tensor compute(Tensor... inputs) {
        return Tensors.convolve(inputs[0], inputs[1], stride);
    }

    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor output, Tensor... inputs) {
        Tensor A = inputs[0]; // input
        Tensor B = inputs[1]; // filter

        int[] aShape = A.shape();
        int[] bShape = B.shape();

        int batch = aShape[0];
        int inChannels = aShape[1];
        int inHeight = aShape[2];
        int inWidth = aShape[3];

        int numFilters = bShape[0];
        int filterHeight = bShape[2];
        int filterWidth = bShape[3];

        int outHeight = gradOutput.shape()[2];
        int outWidth = gradOutput.shape()[3];

        Tensor gradA = Tensors.zerosLike(A);
        Tensor gradB = Tensors.zerosLike(B);

        int batchSize = inChannels * inHeight * inWidth;
        float[] gradAData = gradA.data();

        IntStream.range(0, batch).parallel().forEach(b -> {
            Tensor inputBatch = A.slice(Range.point(b)).squeeze(0); // [C_in, H, W]
            Tensor dOutBatch = gradOutput.slice(Range.point(b)); // [F, H_out, W_out]

            Tensor X_col = Tensors.im2col(inputBatch, filterHeight, filterWidth, stride);
            Tensor W_col = B.reshape(numFilters, inChannels * filterHeight * filterWidth);

            Tensor dY_col = dOutBatch.reshape(numFilters, outHeight * outWidth);
            Tensor dW_col = dY_col.matmul(X_col.transpose());
            gradB.add(dW_col.reshape(B.shape()));

            Tensor dX_col = W_col.transpose().matmul(dY_col);
            Tensor dInputBatch = Tensors.col2im(
                dX_col, inChannels, inHeight, inWidth, filterHeight, filterWidth, stride
            );

            float[] dInputData = dInputBatch.data();
            System.arraycopy(dInputData, 0, gradAData, b * batchSize, batchSize);
        });

        return new Tensor[]{ gradA, gradB };
    }
}
