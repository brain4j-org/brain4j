package net.echo.brain4j.transformers;

import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;

import java.util.*;
import java.util.regex.Pattern;

public class Vocabulary {

    public static final Pattern PATTERN = Pattern.compile("[\\s,;.?!-]+");

    private final List<String> corpus;
    private final List<String> tokens;
    private final int dimension;

    public Vocabulary(List<String> corpus, int dimension) {
        this.corpus = corpus;
        this.dimension = dimension;
        this.tokens = new ArrayList<>();
    }

    public void tokenize() {
        for (String sentence : corpus) {
            String[] tokens = PATTERN.split(sentence);

            System.out.println("TOKENS: " + Arrays.toString(tokens));

            for (String token : tokens) {
                String tok = token.toLowerCase();

                if (!this.tokens.contains(tok)) {
                    this.tokens.add(tok);
                }
            }
        }
    }

    public Tensor encode(String phrase) {
        String[] tokens = PATTERN.split(phrase);

        Tensor result = TensorFactory.zeros(tokens.length, dimension);

        for (int i = 0; i < tokens.length; i++) {
            Tensor encoded = wordToVec(tokens[i]);

            for (int j = 0; j < encoded.elements(); j++) {
                result.set(encoded.get(j), i, j);
            }
        }

        return result;
    }

    public Tensor wordToVec(String word) {
        Tensor result = TensorFactory.zeros(dimension);

        for (int i = 0; i < word.length(); i++) {
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
}
