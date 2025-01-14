package net.echo.brain4j.nlp.model.layers;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.layer.impl.LayerNorm;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.model.initialization.WeightInit;
import net.echo.brain4j.nlp.attention.MultiHeadAttention;
import net.echo.brain4j.training.optimizers.impl.GradientDescent;
import net.echo.brain4j.training.updater.impl.NormalUpdater;
import net.echo.brain4j.utils.Vector;

import java.util.ArrayList;
import java.util.List;

public class TransformerDecoder extends Layer {

    private final Model feedForward;
    private final LayerNorm normalizer;
    private final MultiHeadAttention firstAttention;
    private final MultiHeadAttention secondAttention;

    public TransformerDecoder(int numHeads, int contextSize, int dimension, double temperature) {
        super(0, Activations.LINEAR);
        this.normalizer = new LayerNorm();
        this.firstAttention = new MultiHeadAttention(numHeads, contextSize, dimension, temperature);
        this.secondAttention = new MultiHeadAttention(numHeads, contextSize, dimension, temperature);
        this.feedForward = new Model(
                new DenseLayer(dimension, Activations.LINEAR),
                new DenseLayer(4 * dimension, Activations.RELU),
                new DenseLayer(dimension, Activations.LINEAR)
        );
        this.feedForward.compile(WeightInit.NORMAL, LossFunctions.MEAN_SQUARED_ERROR, new GradientDescent(0.001), new NormalUpdater());
    }

    public List<Vector> transform(List<Vector> embeddings) {
        List<Vector> resulting = new ArrayList<>();

        for (Vector vector : embeddings) {
            Vector embedding = Vector.of(vector.toArray());

            Vector attended = firstAttention.attend(embedding);

            attended.add(vector);
            attended = normalizer.normalize(attended);

            attended = secondAttention.attend(attended);
            attended.add(vector);

            attended = normalizer.normalize(attended);

            Vector result = feedForward.predict(attended);
            result = normalizer.normalize(result);

            resulting.add(result);
        }

        return resulting;
    }
}


