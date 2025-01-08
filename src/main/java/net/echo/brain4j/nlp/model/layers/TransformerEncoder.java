package net.echo.brain4j.nlp.model.layers;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.layer.impl.LayerNorm;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.nlp.attention.MultiHeadAttention;
import net.echo.brain4j.utils.Vector;

import java.util.ArrayList;
import java.util.List;

public class TransformerEncoder extends Layer {

    private final Model feedForward;
    private final LayerNorm normalizer;
    private final MultiHeadAttention attention;

    public TransformerEncoder(int numHeads, int contextSize, int dimension, double temperature) {
        super(0, Activations.LINEAR);
        this.normalizer = new LayerNorm();
        this.attention = new MultiHeadAttention(numHeads, contextSize, dimension, temperature);
        this.feedForward = new Model(
                new DenseLayer(dimension, Activations.LINEAR),
                new DenseLayer(4 * dimension, Activations.RELU),
                new DenseLayer(dimension, Activations.LINEAR)
        );
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


