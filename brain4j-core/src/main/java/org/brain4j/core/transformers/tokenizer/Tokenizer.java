package org.brain4j.core.transformers.tokenizer;

import org.brain4j.core.transformers.Vocabulary;
import org.brain4j.math.tensor.Tensor;

import java.util.List;

public interface Tokenizer {
    List<String> split(String corpus);

    Tensor tokenize(Vocabulary vocabulary, String input);

    String decode(Vocabulary vocabulary, Tensor token);
}
