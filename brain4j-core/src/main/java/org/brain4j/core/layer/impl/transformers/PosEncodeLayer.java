package org.brain4j.core.layer.impl.transformers;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.structure.StatesCache;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;
import org.brain4j.math.tensor.index.Range;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class PosEncodeLayer extends Layer {

    private List<Tensor> encodings;
    private int maxLength;
    private int embeddingDim;

    private PosEncodeLayer() {
    }

    public PosEncodeLayer(int embeddingDim, int maxLength) {
        this.embeddingDim = embeddingDim;
        this.maxLength = maxLength;
        this.encodings = new ArrayList<>();

        initializeEncodings(maxLength);
    }

    public PosEncodeLayer(int embeddingDim) {
        this(embeddingDim, 1024);
    }

    @Override
    public String getLayerName() {
        return "Positional Encoding";
    }

    @Override
    public void serialize(DataOutputStream stream) throws Exception {
        super.serialize(stream);
        stream.writeInt(embeddingDim);
        stream.writeInt(encodings.size());
    }

    @Override
    public void deserialize(DataInputStream stream) throws Exception {
        super.deserialize(stream);
        this.embeddingDim = stream.readInt();
        this.encodings = new ArrayList<>(stream.readInt());
    }

    @Override
    public boolean canPropagate() {
        return false;
    }

    @Override
    public int getTotalWeights() {
        return maxLength * embeddingDim;
    }

    @Override
    public void connect(Random generator, Layer previous, double bound) {
        this.weights = Tensors.zeros(0);
    }

    @Override
    public Tensor forward(StatesCache cache, Tensor input, boolean training) {
        if (input.dimension() < 2) {
            throw new IllegalArgumentException("Input must be at least a 2D matrix!");
        }

        if (input.dimension() != 3) {
            input = input.reshape(1, input.elements());
        }

        int[] shape = input.shape();

        int batchSize = shape[0];
        int seqLength = shape[1];
        int dimension = shape[2];

        Tensor result = Tensors.create(batchSize, seqLength, dimension);

        for (int b = 0; b < batchSize; b++) {
            Tensor batch = input.slice(new Range(b, b + 1))
                    .reshape(seqLength, dimension);

            for (int s = 0; s < seqLength; s++) {
                Tensor token = batch.slice(new Range(s, s + 1))
                        .reshape(1, dimension);
                Tensor encoding = getEncoding(b); // [1, dimension]

                token.add(encoding);

                for (int p = 0; p < token.elements(); p++) {
                    result.set(token.get(0, p), b, s, p);
                }
            }
        }

        return result;
    }

    private void initializeEncodings(int maxLength) {
        for (int position = 0; position < maxLength; position++) {
            encodings.add(generate(position));
        }
    }

    public Tensor generate(int position) {
        Tensor token = Tensors.zeros(embeddingDim);

        for (int i = 0; i < embeddingDim; i++) {
            double exponent = (2.0 * Math.floor(i / 2.0)) / embeddingDim;

            double angle = position / Math.pow(10000, exponent);
            double value = (i % 2 == 0) ? Math.sin(angle) : Math.cos(angle);

            token.set(value, i);
        }

        return token.reshape(1, embeddingDim);
    }

    public Tensor getEncoding(int index) {
        if (index < encodings.size()) {
            return encodings.get(index);
        }

        Tensor token = generate(index);
        encodings.add(token);

        return token;
    }
}
