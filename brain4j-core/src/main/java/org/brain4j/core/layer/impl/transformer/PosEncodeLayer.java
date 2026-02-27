package org.brain4j.core.layer.impl.transformer;

import com.google.gson.JsonObject;
import org.brain4j.core.layer.OldLayer;
import org.brain4j.math.Tensors;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.GpuTensor;
import org.brain4j.math.commons.Range;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class PosEncodeLayer extends OldLayer {

    private final Map<Integer, Tensor> preGenerated = new HashMap<>();
    private int dimension;
    private int length = 5000;

    private PosEncodeLayer() {
    }

    public PosEncodeLayer(int length, int dimension) {
        this.dimension = dimension;
        this.length = length;

        for (int i = 0; i < length; i++) {
            preGenerated.put(i, generate(i, dimension));
        }
    }

    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        validateInputLength(inputs);

        // [batch, seq_len, dimension]
        Tensor input = inputs[0];
        int[] shape = input.shape();

        if (shape.length != 3) {
            throw Commons.illegalArgument("Input must have shape [batch, seq_length, dimension]! Got: %s",
                Arrays.toString(shape));
        }

        int seqLength = shape[1];
        int dimension = shape[2];
        
        Tensor positional = Tensors.zeros(seqLength, dimension);
        float[] posData = positional.data();

        for (int i = 0; i < seqLength; i++) {
            Tensor add = preGenerated.computeIfAbsent(i, index -> generate(index, dimension));
            float[] addData = add.data();

            int index = i * dimension;

            System.arraycopy(addData, 0, posData, index, addData.length);
        }

        Tensor output = input.add(positional);

        if (input instanceof GpuTensor gpuTensor) output = output.to(gpuTensor.device());
        if (input.usesGrad()) output = output.withGrad();

        return new Tensor[] { output };
    }
    
    @Override
    public void serialize(JsonObject object) {
        object.addProperty("dimension", dimension);
        object.addProperty("length", length);
    }
    
    @Override
    public void deserialize(JsonObject object) {
        this.dimension = object.get("dimension").getAsInt();
        this.length = object.get("length").getAsInt();
    }
    
    @Override
    public int size() {
        return dimension;
    }
    
    @Override
    public void setWeights(Tensor weights) {
        this.length = weights.shapeAt(0);
        
        for (int i = 0; i < length; i++) {
            Tensor slice = weights.slice(Range.point(i), Range.all());
            preGenerated.put(i, slice.squeeze());
        }
    }
    
    public Tensor generate(int position, int embeddingDim) {
        Tensor token = Tensors.zeros(embeddingDim);

        for (int i = 0; i < embeddingDim; i++) {
            double exponent = (2.0 * Math.floor(i / 2.0)) / embeddingDim;

            double angle = position / Math.pow(10000, exponent);
            double value = (i % 2 == 0) ? Math.sin(angle) : Math.cos(angle);

            token.set(value, i);
        }

        return token.reshape(1, embeddingDim);
    }
}
