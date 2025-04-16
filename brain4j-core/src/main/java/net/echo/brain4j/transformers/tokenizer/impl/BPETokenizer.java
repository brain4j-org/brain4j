package net.echo.brain4j.transformers.tokenizer.impl;

import net.echo.brain4j.transformers.tokenizer.Tokenizer;
import net.echo.brain4j.transformers.Vocabulary;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;

import java.util.Arrays;
import java.util.List;

public class BPETokenizer implements Tokenizer {

    private final Vocabulary vocab;

    public BPETokenizer(Vocabulary vocab) {
        this.vocab = vocab;
    }

    private List<String> preTokenize(String input) {
        input = input.trim().toLowerCase();
        return Arrays.asList(input.split("\\s+"));
    }

    @Override
    public List<String> split(String input) {
        List<String> result = preTokenize(input);

        // TODO: Apply byte pair encoding here

        return result;
    }

    @Override
    public Tensor encode(List<String> tokens) {
        float[] ids = new float[tokens.size()];

        for (int i = 0; i < tokens.size(); i++) {
            ids[i] = vocab.getId(tokens.get(i));
        }

        return TensorFactory.vector(ids);
    }

    @Override
    public String decode(Tensor tokens) {
        if (tokens.dimension() != 2) {
            throw new IllegalArgumentException("Tensor to decode must be 2D!");
        }

        List<Tensor> split = TensorFactory.toList(tokens);
        StringBuilder result = new StringBuilder();

        for (Tensor token : split) {
            result.append(vocab.getToken(token.argmax()));
        }

        return result.toString();
    }
}
