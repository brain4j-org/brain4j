package org.brain4j.core.model.impl;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.impl.transformer.EmbeddingLayer;
import org.brain4j.core.layer.impl.transformer.OutVocabLayer;
import org.brain4j.core.layer.impl.transformer.PosEncodeLayer;
import org.brain4j.core.layer.impl.transformer.TransformerDecoder;

/**
 * Represents the transformer architecture as described
 * in the "Attention is All You Need" paper.
 *
 * @author xEcho1337
 * @since 3.0
 */
public class Transformer extends Sequential {

    public static Transformer decoderOnly(int vocabSize, int dimension, int layers, int heads, double temperature) {
        Transformer model = new Transformer(
                new EmbeddingLayer(vocabSize, dimension),
                new PosEncodeLayer()
        );

        for (int i = 0; i < layers; i++) {
            model.add(new TransformerDecoder(heads, dimension, 0.1));
        }

        model.add(new OutVocabLayer(vocabSize, dimension, temperature));

        return model;
    }

    /**
     * Constructs a new instance of the transformer architecture with the given layers.
     * @param layers the sequence of layers forming the neural network
     */
    public static Transformer of(Layer... layers) {
        return new Transformer(layers);
    }

    protected Transformer(Layer... layers) {
        super(layers);
    }

    public static Transformer.Builder newBuilder() {
        return new Transformer.Builder();
    }

    public static class Builder extends Sequential.Builder {

        private int vocabSize;
        private int dimension;
        private int heads;
        private boolean hasEmbedding;
        private boolean hasPosEncode;
        private boolean hasOutVocab;
        private double temperature;

        protected Builder() {
            super();
        }

        public Builder setVocabSize(int vocabSize) {
            if (vocabSize < 1) {
                throw new IllegalArgumentException("Vocabulary size must be greater than 0!");
            }

            this.vocabSize = vocabSize;
            return this;
        }

        public Builder setDimension(int dimension) {
            if (dimension < 1) {
                throw new IllegalArgumentException("Dimension size must be greater than 0!");
            }

            this.dimension = dimension;
            return this;
        }

        public Builder addDecoders(int amount, double dropout) {
            if (amount < 1) {
                throw new IllegalArgumentException("Amount must be greater than 0!");
            }

            if (dropout < 0.0 || dropout > 1.0) {
                throw new IllegalArgumentException("Dropout must be between 0 and 1!");
            }

            for (int i = 0; i < amount; i++) {
                this.layers.add(new TransformerDecoder(heads, dimension, dropout));
            }

            return this;
        }

        public Builder setHeads(int heads) {
            if (heads < 1) {
                throw new IllegalArgumentException("Number of heads must be greater than 0!");
            }

            this.heads = heads;
            return this;
        }

        public Builder setHasEmbedding(boolean hasEmbedding) {
            this.hasEmbedding = hasEmbedding;
            return this;
        }

        public Builder setHasPosEncode(boolean hasPosEncode) {
            this.hasPosEncode = hasPosEncode;
            return this;
        }

        public Builder setHasOutVocab(boolean hasOutVocab) {
            this.hasOutVocab = hasOutVocab;
            return this;
        }

        public Builder setTemperature(double temperature) {
            if (temperature < 0.0 || temperature > 1.0) {
                throw new IllegalArgumentException("Temperature must be between 0.0 and 1.0!");
            }

            this.temperature = temperature;
            return this;
        }

        public Transformer compile() {
            if (hasPosEncode) {
                layers.addFirst(new PosEncodeLayer());
            }

            if (hasEmbedding) {
                layers.addFirst(new EmbeddingLayer(vocabSize, dimension));
            }

            if (hasOutVocab) {
                layers.add(new OutVocabLayer(vocabSize, dimension, temperature));
            }

            Transformer model = new Transformer(layers.toArray(new Layer[0]));

            if (optimizer == null || lossFunction == null) {
                throw new IllegalStateException("Optimizer and loss function are both null! Initialize them first.");
            }

            return (Transformer) model.compile(lossFunction, optimizer, updater);
        }
    }
}
