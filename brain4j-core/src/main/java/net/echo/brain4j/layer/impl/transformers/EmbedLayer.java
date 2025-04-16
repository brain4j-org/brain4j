package net.echo.brain4j.layer.impl.transformers;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.StatesCache;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;

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
    public void connect(Random generator, Layer previous, Layer next, double bound) {
        this.weights = TensorFactory.matrix(vocabSize, embeddingDim);

        for (int i = 0; i < vocabSize; i++) {
            for (int j = 0; j < embeddingDim; j++) {
                double value = generator.nextDouble(2 * bound) - bound;
                this.weights.set(value, i, j);
            }
        }

        this.embeddings = TensorFactory.toList(weights);
    }

    @Override
    public Tensor forward(StatesCache cache, Layer lastLayer, Layer nextLayer, Tensor input, boolean training) {
        if (input.dimension() != 1) {
            throw new UnsupportedOperationException("Input must be 1-dimensional!");
        }

        int rows = input.shape()[0];
        List<Tensor> tokens = new ArrayList<>();

        for (int i = 0; i < rows; i++) {
            int index = (int) input.get(i);

            if (index < 0 || index >= vocabSize) {
                throw new IllegalArgumentException("Invalid index: %s for input tensor: %s".formatted(index,
                        input.toString("%.1f")));
            }

            Tensor embedding = embeddings.get(index);
            tokens.add(embedding);
        }

        return TensorFactory.mergeTensors(tokens);
    }
}
