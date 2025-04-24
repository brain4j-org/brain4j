package net.echo.brain4j.transformers.tokenizer.impl;

import net.echo.brain4j.transformers.Vocabulary;
import net.echo.math.tensor.Tensor;

public class BPETokenizer extends SimpleTokenizer {

    @Override
    public Tensor tokenize(Vocabulary vocabulary, String input) {
        throw new UnsupportedOperationException("Not implemented yet for this class.");
    }
}
