package net.echo.brain4j.transformers.tokenizer;

import net.echo.math4j.math.tensor.Tensor;

import java.util.List;

public interface Tokenizer {
    List<String> split(String input);

    Tensor encode(List<String> tokens);

    String decode(Tensor tokens);
}
