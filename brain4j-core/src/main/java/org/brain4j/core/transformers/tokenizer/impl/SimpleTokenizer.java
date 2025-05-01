package org.brain4j.core.transformers.tokenizer.impl;

import org.brain4j.core.transformers.Vocabulary;
import org.brain4j.core.transformers.tokenizer.Tokenizer;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

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

        return Tensors.vector(ids);
    }

    @Override
    public String decode(Vocabulary vocabulary, Tensor token) {
        return vocabulary.getToken(token.argmax());
    }
}
