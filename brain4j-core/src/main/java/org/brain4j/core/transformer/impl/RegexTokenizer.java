package org.brain4j.core.transformer.impl;

import org.brain4j.core.transformer.Tokenizer;

import java.util.ArrayList;
import java.util.Arrays;
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
}
