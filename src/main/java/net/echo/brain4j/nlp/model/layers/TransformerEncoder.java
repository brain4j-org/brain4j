package net.echo.brain4j.nlp.model.layers;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.layer.impl.LayerNorm;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.model.initialization.WeightInit;
import net.echo.brain4j.nlp.attention.MultiHeadAttention;
import net.echo.brain4j.training.optimizers.Optimizer;
import net.echo.brain4j.training.updater.Updater;
import net.echo.brain4j.utils.Vector;

import java.util.ArrayList;
import java.util.List;

public class TransformerEncoder {

    private final int heads;
    private final int contextSize;
    private final int dimension;
    private final double temperature;

    private final Model feedForward;
    private final LayerNorm normalizer;

    private MultiHeadAttention attention;

    public TransformerEncoder(int numHeads, int contextSize, int dimension, double temperature) {
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
        this.attention = new MultiHeadAttention(weightInit, heads, contextSize, dimension, temperature);
    }

    /**
     * Transforms a list of embeddings using a sequence of neural network layers.
     * <p>
     * The transformation is applied in the following order:
     * <ol>
     *     <li>Layer Normalization</li>
     *     <li>Multi-Head Attention</li>
     *     <li>Layer Normalization</li>
     *     <li>Feed Forward Network</li>
     *     <li>Layer Normalization</li>
     * </ol>
     *
     * @param embeddings the list of embeddings to transform
     */
    public List<Vector> transform(List<Vector> embeddings) {
        List<Vector> resulting = new ArrayList<>();

        for (Vector vector : embeddings) {
            Vector embedding = Vector.of(vector.toArray());
            embedding = normalizer.normalize(embedding);

            Vector attended = attention.attend(embedding);
            attended = normalizer.normalize(attended);

            System.out.println("Attended");
            System.out.println(attended);
            Vector result = feedForward.predict(attended);
            // result = normalizer.normalize(result);

            System.out.println("Result");
            System.out.println(result);
            resulting.add(result);
        }

        return resulting;
    }

    public Model getFeedForward() {
        return feedForward;
    }

    public LayerNorm getNormalizer() {
        return normalizer;
    }

    public MultiHeadAttention getAttention() {
        return attention;
    }
}


