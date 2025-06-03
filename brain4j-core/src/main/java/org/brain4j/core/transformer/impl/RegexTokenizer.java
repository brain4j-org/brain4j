package org.brain4j.core.transformer.impl;

import org.brain4j.core.transformer.Tokenizer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class RegexTokenizer implements Tokenizer {

    @Override
    public List<String> tokenize(String input) {
        String[] raw = input.split("(?<=\\p{Punct})|(?=\\p{Punct})|\\s+");

        List<String> tokens = new ArrayList<>();

        for (String t : raw) {
            if (!t.isBlank()) {
                tokens.add(t);
            }
        }

        return tokens;
    }

    @Override
    public void save(String path) throws IOException {

    }

    @Override
    public void load(String path) throws IOException {

    }
}
