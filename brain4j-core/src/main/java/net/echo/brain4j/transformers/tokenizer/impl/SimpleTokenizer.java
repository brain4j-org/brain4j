package net.echo.brain4j.transformers.tokenizer.impl;

import net.echo.brain4j.transformers.Vocabulary;
import net.echo.brain4j.transformers.tokenizer.Tokenizer;
import net.echo.math.tensor.Tensor;
import net.echo.math.tensor.TensorFactory;

import java.util.ArrayList;
import java.util.List;

public class SimpleTokenizer implements Tokenizer {

    @Override
    public List<String> split(String corpus) {
        List<String> tokens = new ArrayList<>();
        StringBuilder current = new StringBuilder();

        for (int i = 0; i < corpus.length(); i++) {
            char c = corpus.charAt(i);

            if (Character.isLetterOrDigit(c)) {
                current.append(c);
            } else {
                if (!current.isEmpty()) {
                    tokens.add(current.toString());
                    current.setLength(0);
                }

                if (c == ' ') {
                    current.append(' ');
                } else {
                    tokens.add(String.valueOf(c));
                }
            }
        }

        if (!current.isEmpty()) {
            tokens.add(current.toString());
        }

        return tokens;
    }

    @Override
    public Tensor tokenize(Vocabulary vocabulary, String input) {
        List<String> tokens = split(input);
        float[] ids = new float[tokens.size()];

        for (int i = 0; i < tokens.size(); i++) {
            ids[i] = vocabulary.getId(tokens.get(i));
        }

        return TensorFactory.vector(ids);
    }

    @Override
    public String decode(Vocabulary vocabulary, Tensor token) {
        return vocabulary.getToken(token.argmax());
    }
}
