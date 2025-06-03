package org.brain4j.core.transformer.tokenizers;

import org.brain4j.math.tensor.Tensor;

import java.io.IOException;
import java.util.List;
import java.util.Map;

public interface Tokenizer {
    List<String> tokenize(String input);

    Tensor encode(List<String> tokens);

    String decode(int index);

    Map<String, Integer> vocab();

    void save(String path) throws IOException;

    void load(String path) throws IOException;
}
