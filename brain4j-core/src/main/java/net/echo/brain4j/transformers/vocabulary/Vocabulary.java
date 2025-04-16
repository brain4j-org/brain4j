package net.echo.brain4j.transformers.vocabulary;

import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Vocabulary {

    private final List<String> corpus;
    private final List<String> tokens;
    private final int dimension;

    public Vocabulary(List<String> corpus, int dimension) {
        this.corpus = corpus;
        this.dimension = dimension;
        this.tokens = new ArrayList<>();
        this.tokens.add("<unk>"); // The first index is for the unknown token
        this.tokens.add("<end>"); // The second index is for the unknown token
        this.tokenize();
    }

    public List<String> split(String input) {
        List<String> result = new ArrayList<>();

        StringBuilder current = new StringBuilder();
        List<Character> delimiters = Arrays.asList(' ', ',', ';', '.', '?', '!', '-', '_', '|');

        for (char c : input.toCharArray()) {
            if (delimiters.contains(c)) {
                String charAsString = String.valueOf(c);

                if (!current.isEmpty()) {
                    result.add(current.toString());
                    current = new StringBuilder();
                }

                result.add(charAsString);
            } else {
                current.append(c);
            }
        }

        if (!current.isEmpty()) {
            result.add(current.toString());
        }

        return result;
    }

    public void tokenize() {
        for (String token : corpus) {
            List<String> split = split(token);

            for (String s : split) {
                if (tokens.contains(s)) continue;

                tokens.add(s);
            }
        }
    }

    public Tensor encode(String phrase) {
        List<String> tokens = split(phrase);
        Tensor result = TensorFactory.zeros(tokens.size(), dimension);

        for (int i = 0; i < tokens.size(); i++) {
            Tensor encoded = wordToVec(tokens.get(i));

            for (int j = 0; j < encoded.elements(); j++) {
                result.set(encoded.get(j), i, j);
            }
        }

        return result;
    }

    public Tensor wordToVec(String word) {
        Tensor result = TensorFactory.zeros(dimension);

        for (int i = 0; i < Math.min(word.length(), dimension); i++) {
            char c = word.charAt(i);
            result.set(Math.sin(c), i);
        }

        return result;
    }

    public String indexToWord(int index) {
        return tokens.get(index);
    }

    public int getVocabSize() {
        return tokens.size();
    }

    public int wordToIndex(String expected) {
        int index = tokens.indexOf(expected);

        if (index == -1) {
            tokens.add(expected);
            return wordToIndex(expected);
        }

        return index;
    }
}
