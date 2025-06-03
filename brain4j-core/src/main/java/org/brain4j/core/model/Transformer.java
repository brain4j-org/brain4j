package org.brain4j.core.model;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.impl.transformer.EmbeddingLayer;
import org.brain4j.core.layer.impl.transformer.OutVocabulary;
import org.brain4j.core.layer.impl.transformer.PosEncodeLayer;
import org.brain4j.core.layer.impl.transformer.TransformerDecoder;

/**
 * Represents the transformer architecture as described
 * in the "Attention is All You Need" paper.
 * @since 3.0
 * @author xEcho1337
 */
public class Transformer extends Model {

    public static Transformer decoderOnly(int vocabSize, int dimension, int layers, int heads, double temperature) {
        Transformer model = new Transformer(
                new EmbeddingLayer(vocabSize, dimension),
                new PosEncodeLayer()
        );

        for (int i = 0; i < layers; i++) {
            model.add(new TransformerDecoder(heads, dimension, 0.1));
        }

        model.add(new OutVocabulary(vocabSize, dimension, temperature));

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
}
