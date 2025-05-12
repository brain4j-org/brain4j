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

public class EmbedLayer extends Layer {

    private List<Tensor> embeddings;
    private int vocabSize;
    private int embeddingDim;

    private EmbedLayer() {
    }

    public EmbedLayer(int vocabSize, int embeddingDim) {
        this.vocabSize = vocabSize;
        this.embeddingDim = embeddingDim;
    }

    @Override
    public String getLayerName() {
        return "Embedding";
    }

    @Override
    public void serialize(DataOutputStream stream) throws Exception {
        super.serialize(stream);
        stream.writeInt(vocabSize);
        stream.writeInt(embeddingDim);
    }

    @Override
    public void deserialize(DataInputStream stream) throws Exception {
        super.deserialize(stream);
        this.vocabSize = stream.readInt();
        this.embeddingDim = stream.readInt();
        this.embeddings = Tensors.toList(weights);
    }

    @Override
    public void connect(
        Random generator,
        Layer previous,
        double bound
    ) {
        this.weights = Tensors.matrix(vocabSize, embeddingDim);

        for (int i = 0; i < vocabSize; i++) {
            for (int j = 0; j < embeddingDim; j++) {
                double value = generator.nextDouble(2) - 1;
                this.weights.set(value, i, j);
            }
        }

        this.embeddings = Tensors.toList(weights);
    }

    @Override
    public Tensor forward(
        int index,
        StatesCache cache,
        Tensor input,
        boolean training
    ) {
        if (input.dimension() > 2) {
            throw new IllegalArgumentException("Input must be a 1D or 2D matrix!");
        }

        if (input.dimension() == 1) {
            input = input.reshape(1, input.elements());
        }

        int[] shape = input.shape();

        int batchSize = shape[0];
        int elements = shape[1];

        Tensor result = Tensors.create(batchSize, elements, embeddingDim);

        for (int i = 0; i < batchSize; i++) {
            Tensor batch = input.slice(new Range(i, i + 1)); // [1, seq_len]
            List<Tensor> tokens = new ArrayList<>();

            for (int j = 0; j < elements; j++) {
                int batchIndex = (int) batch.get(0, j);

                if (batchIndex < 0 || batchIndex >= vocabSize) {
                    throw new IllegalArgumentException(
                            "Invalid index: " + batchIndex + " for input tensor: " + input.toString("%.1f")
                    );
                }

                Tensor embedding = embeddings.get(batchIndex);
                tokens.add(embedding);
            }

            result.setChannel(i, Tensors.mergeTensors(tokens));
        }

        return result;
    }
}
