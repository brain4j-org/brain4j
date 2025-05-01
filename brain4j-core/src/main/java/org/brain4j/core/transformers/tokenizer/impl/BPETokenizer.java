package org.brain4j.core.transformers.tokenizer.impl;

import org.brain4j.core.transformers.Vocabulary;
import org.brain4j.math.tensor.Tensor;

public class BPETokenizer extends SimpleTokenizer {

    @Override
    public Tensor tokenize(Vocabulary vocabulary, String input) {
        throw new UnsupportedOperationException("Not implemented yet for this class.");
    }
}
