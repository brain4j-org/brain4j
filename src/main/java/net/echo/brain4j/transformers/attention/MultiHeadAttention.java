package net.echo.brain4j.transformers.attention;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.model.initialization.WeightInit;
import net.echo.brain4j.utils.Vector;

import java.util.ArrayList;
import java.util.List;

public class MultiHeadAttention extends Layer<Vector, Vector> {

    private final List<AttentionHead> heads;
    private final WeightInit weightInit;
    private final double temperature;
    private final int headCount;
    private final int contextSize;
    private final int dimension;

    private Vector input;
    private Vector output;

    public MultiHeadAttention(WeightInit weightInit, int headCount, int contextSize, int dimension, double temperature) {
        super(0, Activations.LINEAR);

        this.heads = new ArrayList<>();
        this.weightInit = weightInit;
        this.headCount = headCount;
        this.contextSize = contextSize;
        this.dimension = dimension;
        this.temperature = temperature;

        initializeHeads();
    }

    public Vector attend(Vector input) {
        List<Vector> attendedChanges = new ArrayList<>();

        for (AttentionHead head : heads) {
            attendedChanges.add(head.attend(input));
        }

        Vector result = input.clone();

        for (Vector changes : attendedChanges) {
             result.add(changes);
        }

        return result;
    }

    private void initializeHeads() {
        for (int i = 0; i < headCount; i++) {
            heads.add(new AttentionHead(weightInit, contextSize, dimension, temperature));
        }
    }
}
