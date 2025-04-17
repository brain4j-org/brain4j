package net.echo.brain4j.transformers.tokenizer;

import net.echo.brain4j.transformers.Vocabulary;
import net.echo.math.tensor.Tensor;

import java.util.List;

public interface Tokenizer {
    List<String> split(String corpus);

    Tensor tokenize(Vocabulary vocabulary, String input);

    String decode(Vocabulary vocabulary, Tensor token);
}
