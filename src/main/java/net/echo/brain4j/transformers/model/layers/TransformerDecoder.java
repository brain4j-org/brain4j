package net.echo.brain4j.transformers.model.layers;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.layer.impl.LayerNorm;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.model.initialization.WeightInit;
import net.echo.brain4j.transformers.attention.MultiHeadAttention;
import net.echo.brain4j.training.optimizers.Optimizer;
import net.echo.brain4j.training.updater.Updater;
import net.echo.brain4j.utils.Vector;

import java.util.ArrayList;
import java.util.List;

public class TransformerDecoder {

    private final int heads;
    private final int contextSize;
    private final int dimension;
    private final double temperature;

    private final Model feedForward;
    private final LayerNorm normalizer;

    private MultiHeadAttention firstAttention;
    private MultiHeadAttention secondAttention;

    public TransformerDecoder(int numHeads, int contextSize, int dimension, double temperature) {
        this.heads = numHeads;
        this.contextSize = contextSize;
        this.dimension = dimension;
        this.temperature = temperature;

        this.normalizer = new LayerNorm();
        this.feedForward = new Model(
                new DenseLayer(dimension, Activations.LINEAR),
                new DenseLayer(4 * dimension, Activations.GELU),
                new DenseLayer(dimension, Activations.LINEAR)
        );
    }

    public void compile(WeightInit weightInit, LossFunctions lossFunction, Optimizer optimizer, Updater updater) {
        this.feedForward.compile(weightInit, lossFunction, optimizer, updater);
        this.firstAttention = new MultiHeadAttention(weightInit, heads, contextSize, dimension, temperature);
        this.secondAttention = new MultiHeadAttention(weightInit, heads, contextSize, dimension, temperature);
    }

    public List<Vector> transform(List<Vector> embeddings) {
        List<Vector> resulting = new ArrayList<>();

        for (Vector vector : embeddings) {
            Vector embedding = Vector.of(vector.toArray());

            Vector attended = firstAttention.attend(embedding);
            System.out.println("FIRST ATTENTION: " + attended);

            attended.add(vector);
            System.out.println("FIRST ADD: " + attended);
            attended = normalizer.normalize(attended);
            System.out.println("FIRST NORMALIZE: " + attended);

            attended = secondAttention.attend(attended);
            System.out.println("SECOND ATTENTION: " + attended);
            attended.add(vector);
            System.out.println("SECOND ADD: " + attended);

            attended = normalizer.normalize(attended);
            System.out.println("SECOND NORMALIZE: " + attended);

            Vector result = feedForward.predict(attended);
            System.out.println("FEED FORWARD: " + result);
            result = normalizer.normalize(result);
            System.out.println("NORMALIZE: " + result);

            resulting.add(result);
        }

        return resulting;
    }
}


