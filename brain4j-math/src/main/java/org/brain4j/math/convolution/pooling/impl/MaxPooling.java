package org.brain4j.math.convolution.pooling.impl;

import org.brain4j.math.Tensors;
import org.brain4j.math.convolution.pooling.PoolingProvider;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.SiliconGpuTensor;

import java.util.stream.IntStream;

public class MaxPooling extends PoolingProvider {

    private int[] xCoords;
    private int[] yCoords;
    private int outHeight;
    private int outWidth;

    public MaxPooling(int stride, int windowHeight, int windowWidth) {
        super(stride, windowHeight, windowWidth);
    }
    
    @Override
    public Tensor poolCPU(Tensor input) {
        int[] shape = input.shape();
        int rank = input.rank();
        
        if (rank < 2) {
            throw new IllegalArgumentException("Pooling requires at least 2D tensor.");
        }
        
        int inHeight = shape[rank - 2];
        int inWidth = shape[rank - 1];
        this.outHeight = (inHeight - windowHeight) / stride + 1;
        this.outWidth = (inWidth - windowWidth) / stride + 1;
        
        int outerSize = 1;
        for (int i = 0; i < rank - 2; i++) outerSize *= shape[i];
        
        float[] inputData = input.data();
        float[] outputData = new float[outerSize * outHeight * outWidth];
        
        this.xCoords = new int[outerSize * outHeight * outWidth];
        this.yCoords = new int[outerSize * outHeight * outWidth];
        IntStream.range(0, outerSize).parallel().forEach(outer -> {
            int offsetIn = outer * inHeight * inWidth;
            int offsetOut = outer * outHeight * outWidth;
            pool2D(inputData, outputData, offsetIn, offsetOut, inWidth);
        });
        
        int[] outShape = new int[rank];
        System.arraycopy(shape, 0, outShape, 0, rank - 2);
        outShape[rank - 2] = outHeight;
        outShape[rank - 1] = outWidth;
        
        return Tensors.create(outShape, outputData);
    }
    
    @Override
    public Tensor poolGPU(Tensor input) {
        return null; // TODO
    }
    
    @Override
    public Tensor backwardCPU(Tensor gradOutput, Tensor input) {
        int[] shape = input.shape();
        int rank = shape.length;

        int inHeight = shape[rank - 2];
        int inWidth = shape[rank - 1];

        int outerSize = 1;
        for (int i = 0; i < rank - 2; i++) outerSize *= shape[i];

        float[] gradIn = new float[input.elements()];
        float[] gradOut = gradOutput.data();

        for (int outer = 0; outer < outerSize; outer++) {
            int offsetIn = outer * inHeight * inWidth;
            int offsetOut = outer * outHeight * outWidth;
            backward2D(gradIn, gradOut, offsetIn, offsetOut, inWidth);
        }

        return Tensors.create(shape, gradIn);
    }
    
    @Override
    public Tensor backwardGPU(SiliconGpuTensor gradient, SiliconGpuTensor input) {
        return null; // TODO
    }
    
    private void pool2D(float[] in, float[] out, int offsetIn, int offsetOut, int inW) {
        int outIdx = offsetOut;
        
        for (int h = 0; h < outHeight; h++) {
            int hStart = h * stride;
            for (int w = 0; w < outWidth; w++) {
                int wStart = w * stride;
                int inBase = offsetIn + hStart * inW + wStart;
                
                float maxValue = Float.NEGATIVE_INFINITY;
                int maxRel = 0;
                
                for (int kh = 0, base = inBase; kh < windowHeight; kh++, base += inW) {
                    int idx = base;
                    for (int kw = 0; kw < windowWidth; kw++, idx++) {
                        float value = in[idx];
                        
                        if (value > maxValue) {
                            maxValue = value;
                            maxRel = kh * inW + kw;
                        }
                    }
                }
                
                out[outIdx] = maxValue;
                int abs = inBase + maxRel;
                yCoords[outIdx] = (abs - offsetIn) / inW;
                xCoords[outIdx] = (abs - offsetIn) % inW;
                outIdx++;
            }
        }
    }
    
    private void backward2D(float[] gradIn, float[] gradOut, int offIn, int offOut, int inW) {
        for (int i = 0; i < outHeight * outWidth; i++) {
            int y = yCoords[offOut + i];
            int x = xCoords[offOut + i];
            gradIn[offIn + y * inW + x] += gradOut[offOut + i];
        }
    }
}