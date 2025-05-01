package org.brain4j.core.layer.impl.transformers;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.structure.StatesCache;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class EmbedLayer extends Layer {

    private final int vocabSize;
    private final int embeddingDim;
    private List<Tensor> embeddings;

    public EmbedLayer(int vocabSize, int embeddingDim) {
        this.vocabSize = vocabSize;
        this.embeddingDim = embeddingDim;
    }

    @Override
    public void connect(Random generator, Layer previous, double bound) {
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
    public Tensor forward(StatesCache cache, Layer lastLayer, Tensor input, boolean training) {
        int elements = input.shape()[0];

        if (input.dimension() > 1) {
            input = input.reshape(input.elements());
        }

        List<Tensor> tokens = new ArrayList<>();

        for (int i = 0; i < elements; i++) {
            int index = (int) input.get(i);

            if (index < 0 || index >= vocabSize) {
                throw new IllegalArgumentException(
                        "Invalid index: " + index + " for input tensor: " + input.toString("%.1f"));
            }

            Tensor embedding = embeddings.get(index);
            tokens.add(embedding);
        }

        return Tensors.mergeTensors(tokens);
    }
}
