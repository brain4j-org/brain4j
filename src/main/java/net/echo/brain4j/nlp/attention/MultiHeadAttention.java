package net.echo.brain4j.nlp.attention;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.utils.Vector;

import java.util.ArrayList;
import java.util.List;

public class MultiHeadAttention extends Layer {

    private final List<AttentionHead> heads;
    private final double temperature;
    private final int headCount;
    private final int contextSize;
    private final int dimension;

    private Vector input;
    private Vector output;

    public MultiHeadAttention(int headCount, int contextSize, int dimension, double temperature) {
        super(0, Activations.LINEAR);

        this.heads = new ArrayList<>();
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
            result = result.add(changes);
        }

        return result;
    }

    private void initializeHeads() {
        for (int i = 0; i < headCount; i++) {
            heads.add(new AttentionHead(contextSize, dimension, temperature));
        }
    }
}
