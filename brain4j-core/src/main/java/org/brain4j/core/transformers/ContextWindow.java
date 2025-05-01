package org.brain4j.core.transformers;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

import java.util.ArrayList;
import java.util.List;

public class ContextWindow {

    private final List<Tensor> tokens;
    private Tensor context;
    private int maxLength;
    private int embeddingDimension;

    public ContextWindow(int maxLength, int embeddingDimension) {
        this.tokens = new ArrayList<>();
        this.maxLength = maxLength;
        this.embeddingDimension = embeddingDimension;
    }

    public void append(Tensor token) {
        while (tokens.size() >= maxLength) {
            tokens.removeFirst();
        }

        if (token.dimension() == 1) {
            token = token.transpose();
        }

        this.tokens.add(token);
        this.context = Tensors.mergeTensors(tokens);
    }

    public void setContext(Tensor context) {
        if (context == null) {
            throw new IllegalArgumentException("Context tensor cannot be null.");
        }

        if (context.dimension() == 1) {
            context = context.transpose();
        }

        this.tokens.clear();
        this.tokens.addAll(Tensors.toList(context));
        this.context = context;
    }

    public Tensor toInput() {
        return context;
    }

    public int getMaxLength() {
        return maxLength;
    }

    public void setMaxLength(int maxLength) {
        this.maxLength = maxLength;
    }

    public int getEmbeddingDimension() {
        return embeddingDimension;
    }

    public void setEmbeddingDimension(int embeddingDimension) {
        this.embeddingDimension = embeddingDimension;
    }
}
